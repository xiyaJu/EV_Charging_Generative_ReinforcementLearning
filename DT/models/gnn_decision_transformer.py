import numpy as np
import torch
import torch.nn as nn

import transformers

from DT.models.model import TrajectoryModel
from DT.models.trajectory_gpt2 import GPT2Model
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

from DT.models.graph_components import GraphStateEncoder


class GNN_DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            remove_act_embs=False,
            fx_node_sizes={},
            feature_dim=8,
            GNN_hidden_dim=32,
            num_gcn_layers=3,
            action_masking=False,
            config=None,
            device=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.config = config
        self.action_masking = action_masking
        self.device = device
        self.hidden_size = hidden_size
        self.fx_node_sizes = fx_node_sizes
        self.feature_dim = feature_dim
        self.GNN_hidden_dim = GNN_hidden_dim
        self.num_gcn_layers = num_gcn_layers

        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(gpt_config)
        # gpt_config = transformers.GPTNeoConfig(
        #     vocab_size=1,  # doesn't matter -- we don't use the vocab
        #     hidden_size=hidden_size
        # )
        # self.transformer = GPTNeoForCausalLM(gpt_config)
        # self.transformer = MambaForCausalLM(gpt_config)

        self.graph_encoder = GraphStateEncoder(
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=GNN_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        mlp_layer_features = hidden_size

        # print("hiden size: ", hidden_size,max_ep_len)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(mlp_layer_features, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [nn.Sigmoid()]))
        )
        self.predict_ev_action = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        # self.actionSigmoid = nn.Sigmoid()
        self.action_tanh = nn.Tanh()
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps,
                attention_mask=None,
                action_mask=None,
                config=None,
                graph_states=None):

        if config is None:
            config = self.config

        if self.action_masking:
            action_mask = action_mask.to(dtype=torch.float32)
            actions = torch.mul(actions, action_mask)

        # input() # pause
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long)

        if graph_states is None:
            states_numpy = states.detach().cpu().numpy()
            graph_states = [[states_numpy[batch, t] for t in range(seq_length)] for batch in range(batch_size)]

        pooled_embedding, x_gnn, gnn_states = self.graph_encoder(
            graph_states,
            config=config,
            batch_size=batch_size,
            seq_length=seq_length,
        )
        # embed each modality with a different head
        state_embeddings = self.embed_state(pooled_embedding)
        # print(f"State embeddings shape: {state_embeddings.shape}")
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        # returns_embeddings = self.embed_return(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3,
                      self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # predict next return given state and action
        return_preds = self.predict_return(x[:, 2])
        # predict next state given state and action
        state_preds = self.predict_state(x[:, 2])
        # predict next action given state
        decision_tokens = x[:, 1].reshape(batch_size * seq_length, self.hidden_size)
        sample_action_preds = []
        ev_node_cursor = 0
        flat_sample_index = 0
        for _ in range(batch_size):
            for _ in range(seq_length):
                num_evs = gnn_states.ev_indexes_node_length[flat_sample_index]
                sample_action_mapper = gnn_states.action_mapper[ev_node_cursor: ev_node_cursor + num_evs]
                sample_action = torch.zeros(
                    self.act_dim,
                    device=decision_tokens.device,
                    dtype=decision_tokens.dtype,
                )
                if num_evs > 0:
                    sample_ev_node_indexes = gnn_states.ev_indexes[ev_node_cursor: ev_node_cursor + num_evs]
                    sample_ev_embeddings = x_gnn[sample_ev_node_indexes]
                    repeated_token = decision_tokens[flat_sample_index].unsqueeze(0).expand(num_evs, -1)
                    ev_action_logits = self.predict_ev_action(
                        torch.cat([sample_ev_embeddings, repeated_token], dim=-1)
                    ).squeeze(-1)
                    sample_action = sample_action.scatter(0, sample_action_mapper, ev_action_logits)
                sample_action_preds.append(sample_action)
                ev_node_cursor += num_evs
                flat_sample_index += 1

        action_preds_t = torch.stack(sample_action_preds, dim=0).reshape(
            batch_size, seq_length, self.act_dim
        )
        # Keep a valid gradient path even when a sampled batch contains no active EV nodes.
        action_preds_t = action_preds_t + 0.0 * decision_tokens.sum()
                
        # pass through tanh to get action values between -1 and 1
        # action_preds_t = self.predict_action(action_preds_t)
        # action_preds = self.predict_action(x[:, 1])
        action_preds_t = self.action_tanh(action_preds_t)

        # print(f"Action preds shape: {action_preds_t.shape}")

        # exit()
        return state_preds, action_preds_t, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps,
                   action_mask,
                   config=None,
                   graph_states=None,
                   **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        action_mask = action_mask.reshape(1, -1, self.act_dim)
        rewards = rewards.reshape(1, -1, 1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            action_mask = action_mask[:, -self.max_length:]
            if graph_states is not None:
                graph_states = list(graph_states)[-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1],
                             self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length -
                             rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length -
                             returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length -
                             timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            action_mask = torch.cat(
                [torch.zeros((action_mask.shape[0], self.max_length -
                             action_mask.shape[1], self.act_dim), device=action_mask.device), action_mask],
                dim=1).to(dtype=torch.float32)
            if graph_states is not None:
                graph_states = [None] * (self.max_length - len(graph_states)) + list(graph_states)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask,
            action_mask=action_mask, config=config, graph_states=graph_states, **kwargs)

        return action_preds[0, -1]
