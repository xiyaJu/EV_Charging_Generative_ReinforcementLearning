import numpy as np
import torch
import torch.nn as nn

import transformers

from DT.models.model import TrajectoryModel
from DT.models.trajectory_gpt2 import GPT2Model
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

from utils import PST_V2G_ProfitMax_state_to_GNN
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, TAGConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class GNN_act_emb_DecisionTransformer(TrajectoryModel):

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
            act_GNN_hidden_dim=32,
            num_act_gcn_layers=3,
            num_gcn_layers=3,            
            action_masking=False,
            config=None,
            device=None,
            gnn_type='GCN',
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
        self.num_act_gcn_layers = num_act_gcn_layers
        self.act_GNN_hidden_dim = act_GNN_hidden_dim

        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(gpt_config)

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        # GCN layers to extract features with a unified edge index
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)

        if num_gcn_layers == 3:
            
            print(f"GNN type: {gnn_type}")
            if gnn_type == 'GCN':
                self.gcn_layers = nn.ModuleList(
                [GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                    # GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim)])
                    GCNConv(2*GNN_hidden_dim, hidden_size)])
            elif gnn_type == 'GAT':
                self.gcn_layers = nn.ModuleList(
                    [GATConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                        GATConv(2*GNN_hidden_dim, hidden_size)])
            elif gnn_type == 'TagConv':
                self.gcn_layers = nn.ModuleList(
                    [TAGConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                        TAGConv(2*GNN_hidden_dim, hidden_size)])
            
            
            mlp_layer_features = hidden_size

        elif num_gcn_layers == 4:
            self.gcn_layers = nn.ModuleList([GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                                             GCNConv(2*GNN_hidden_dim,
                                                     3*GNN_hidden_dim),
                                             GCNConv(3*GNN_hidden_dim, hidden_size)])
            mlp_layer_features = hidden_size

        elif num_gcn_layers == 5:
            self.gcn_layers = nn.ModuleList([GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                                             GCNConv(2*GNN_hidden_dim,
                                                     3*GNN_hidden_dim),
                                             GCNConv(3*GNN_hidden_dim,
                                                     4*GNN_hidden_dim),
                                             GCNConv(4*GNN_hidden_dim,
                                                     hidden_size)
                                             ])
            mlp_layer_features = hidden_size
        else:
            raise ValueError(
                f"Number of GCN layers not supported, use 3, 4, or 5!")

        if num_act_gcn_layers == 3:
            self.act_gcn_layers = nn.ModuleList(
                [GCNConv(1, act_GNN_hidden_dim),
                    GCNConv(act_GNN_hidden_dim, 2*act_GNN_hidden_dim),
                    GCNConv(2*act_GNN_hidden_dim, hidden_size)])
            act_mlp_layer_features = hidden_size

        # print("hiden size: ", hidden_size,max_ep_len)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(mlp_layer_features, hidden_size)
        self.embed_action = torch.nn.Linear(act_mlp_layer_features, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [nn.Sigmoid()]))
        # )
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        # self.actionSigmoid = nn.Sigmoid()
        self.action_tanh = nn.Tanh()
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps,
                attention_mask=None,
                action_mask=None):

        if self.action_masking:
            action_mask = action_mask.to(dtype=torch.float32)
            actions = torch.mul(actions, action_mask)

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long)

        # Convert states to NumPy all at once outside the loops
        states_numpy = states.detach().cpu().numpy()

        gnn_states = [
            [
                PST_V2G_ProfitMax_state_to_GNN(
                    states_numpy[batch, t], self.config)
                for t in range(states.shape[1])
            ]
            for batch in range(states.shape[0])
        ]

        gnn_states = to_GNN_Batch(gnn_states, device=self.device)

        # GNN forward pass
        ev_features = gnn_states.ev_features
        cs_features = gnn_states.cs_features
        tr_features = gnn_states.tr_features
        env_features = gnn_states.env_features
        edge_index = gnn_states.edge_index

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + \
            tr_features.shape[0] + env_features.shape[0]

        embedded_x = torch.zeros(total_nodes,
                                 self.feature_dim,
                                 device=self.device).float()

        # Apply embeddings to the corresponding segments
        if len(gnn_states.ev_indexes) != 0:
            embedded_x[gnn_states.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[gnn_states.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[gnn_states.tr_indexes] = self.tr_embedding(tr_features)

        embedded_x[gnn_states.env_indexes] = self.env_embedding(env_features)

        embedded_x = embedded_x.reshape(-1, self.feature_dim)
        embedded_x = F.relu(embedded_x)

        # Apply GCN layers with the unified edge index
        x_gnn = self.gcn_conv(embedded_x, edge_index)
        x_gnn = F.relu(x_gnn)

        for layer in self.gcn_layers:
            x_gnn = layer(x_gnn, edge_index)
            if layer != self.gcn_layers[-1]:
                x_gnn = F.relu(x_gnn)

        # make batch sample mask
        sample_node_length = gnn_states.sample_node_length
        batch = np.repeat(np.arange(len(sample_node_length)),
                          sample_node_length)
        batch = torch.from_numpy(batch).to(device=self.device)
        batch = batch.to(torch.int64)

        # Graph Embedding
        pooled_embedding = global_mean_pool(x_gnn, batch=batch)


        action_vector = actions.reshape(-1, self.act_dim)
        # print(f"action mapper: {gnn_states.action_mapper}")
        # print(f'gnn_states.ev_indexes_node_length: {gnn_states.ev_indexes_node_length}')
        
        action_features =[]
        edge_indexes_from = []
        edge_indexes_to = []
        node_counter = 0
        act_batch =[]
        
        # if len(gnn_states.ev_indexes_node_length) == 0:
        #     node_counter += len(action_mapper)    
        #     action_features.append(torch.zeros(1,device=self.device))
        #     act_batch.append([i])
        
        for i in range(len(gnn_states.ev_indexes_node_length)):
            action_mapper = gnn_states.action_mapper[
                i: i + gnn_states.ev_indexes_node_length[i]]
            # print(f"Action mapper: {action_mapper}")     
            # print(f"Actions: {action_vector[i, action_mapper]}")
            if len(action_mapper) == 0:
                action_mapper = [0]
                node_counter += len(action_mapper)    
                action_features.append(torch.zeros(1,device=self.device))
                act_batch.append([i])
                edge_indexes_from.append(node_counter)
                edge_indexes_to.append(node_counter)
                continue
                
            # make a fully conntections between all actions
            act_batch.append(np.ones(len(action_mapper))*i)
            for j in range(len(action_mapper)):
                for k in range(len(action_mapper)):    
             
                    edge_indexes_from.append(j + node_counter)
                    edge_indexes_to.append(k + node_counter)
                    

            # print(f"Edge indexes from: {edge_indexes_from_i}")
            # print(f"Edge indexes to: {edge_indexes_to_i}")
            # edge_indexes_from.append(edge_indexes_from_i)
            # edge_indexes_to.append(edge_indexes_to_i)
            node_counter += len(action_mapper)                    
            action_features.append(action_vector[i, action_mapper])
        
        action_features = torch.cat(action_features, dim=0)
        action_features = action_features.unsqueeze(1)
        edge_indexes_from = torch.tensor(edge_indexes_from, device=self.device)        
        edge_indexes_to = torch.tensor(edge_indexes_to, device=self.device)
        act_edge_index = torch.stack([edge_indexes_from, edge_indexes_to], dim=0)
             
        for layer in self.act_gcn_layers:
            action_features = layer(action_features, act_edge_index)
            if layer != self.act_gcn_layers[-1]:
                action_features = F.relu(action_features)
        
        
        #flatten act_batch and make torch
        act_batch = np.concatenate(act_batch)
        act_batch = torch.tensor(act_batch, device=self.device)
        # make int64
        act_batch = act_batch.to(dtype=torch.long)
        # print(f"act batch: {act_batch}")
        # print(f"Act batch shape: {act_batch.shape}")
                
        # print(f"Action features shape: {action_features.shape}")
        # print(f'Actions shape: {actions.shape}')
        
        pooled_embedding_act = global_mean_pool(action_features,
                                                batch=act_batch)
        
        # print(f"Pooled embedding act shape: {pooled_embedding_act.shape}")
        
        # print(f"Pooled embedding shape: {pooled_embedding.shape}")
        # reshape to (batch_size, seq_length, hidden_size)
        pooled_embedding = pooled_embedding.reshape(
            batch_size, seq_length, -1)
        pooled_embedding_act = pooled_embedding_act.reshape(
            batch_size, seq_length, -1)
        
        # embed each modality with a different head
        state_embeddings = self.embed_state(pooled_embedding)
        # print(f"State embeddings shape: {state_embeddings.shape}")
        action_embeddings = self.embed_action(pooled_embedding_act)
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
        # state_preds = self.predict_state(x[:, 2])
        # predict next action given state
        ev_node_features = x_gnn[gnn_states.ev_indexes].permute(1, 0)
        action_preds_t = torch.zeros((batch_size, seq_length, self.act_dim),
                                     device=self.device)

        counter = 0
        node_counter = 0
        for i in range(batch_size):
            for k in range(seq_length):
                action_mapper = gnn_states.action_mapper[
                    node_counter: node_counter + gnn_states.ev_indexes_node_length[counter]]

                action_decoder = x[:, 1][i, k, :].unsqueeze(0)
                action_preds_t[i, k, action_mapper] = torch.matmul(action_decoder, ev_node_features[:,
                                                                                                    node_counter: node_counter + gnn_states.ev_indexes_node_length[counter]])

                node_counter += gnn_states.ev_indexes_node_length[counter]
                counter += 1

        # pass through tanh to get action values between -1 and 1
        # action_preds_t = self.predict_action(action_preds_t)
        # action_preds = self.predict_action(x[:, 1])
        action_preds_t = self.action_tanh(action_preds_t)

        # print(f"Action preds shape: {action_preds_t.shape}")

        # exit()
        return None, action_preds_t, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps,
                   action_mask,
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
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask,
            action_mask=action_mask, **kwargs)

        return action_preds[0, -1]


def to_GNN_Batch(state_batch, device=None):
    # flatten the batch from list of lists of states to a list of states
    states = [state for sublist in state_batch for state in sublist]
    # print(f"States: {states}")
    # print(f"States length: {len(states)}")
    l = len(states)

    edge_index = []
    ev_indexes = np.array([])
    cs_indexes = np.array([])
    tr_indexes = np.array([])
    env_indexes = np.array([])
    action_mapper = np.array([])

    edge_counter = 0
    node_counter = 0

    ev_features = np.concatenate(
        [states[i].ev_features for i in range(l)], axis=0)
    cs_features = np.concatenate(
        [states[i].cs_features for i in range(l)], axis=0)
    tr_features = np.concatenate(
        [states[i].tr_features for i in range(l)], axis=0)
    env_features = np.concatenate(
        [states[i].env_features for i in range(l)], axis=0)
    node_types = np.concatenate(
        [states[i].node_types for i in range(l)], axis=0)
    action_mapper = np.concatenate(
        [states[i].action_mapper for i in range(l)], axis=0)

    sample_node_length = [len(states[i].node_types) for i in range(l)]
    ev_indexes_node_length = [len(states[i].ev_indexes) for i in range(l)]

    for i in range(l):
        edge_index.append(states[i].edge_index + edge_counter)
        ev_indexes = np.concatenate(
            [ev_indexes, states[i].ev_indexes + node_counter], axis=0)
        cs_indexes = np.concatenate(
            [cs_indexes, states[i].cs_indexes + node_counter], axis=0)
        tr_indexes = np.concatenate(
            [tr_indexes, states[i].tr_indexes + node_counter], axis=0)
        env_indexes = np.concatenate(
            [env_indexes, states[i].env_indexes + node_counter], axis=0)

        node_counter += len(states[i].node_types)
        if states[i].edge_index.shape[1] > 0:
            edge_counter += np.max(states[i].edge_index)
        else:
            edge_counter += 1

    edge_index = np.concatenate(edge_index, axis=1)

    state_batch = Data(edge_index=torch.from_numpy(edge_index).to(device),
                       ev_features=torch.from_numpy(
        ev_features).to(device).float(),
        cs_features=torch.from_numpy(
        cs_features).to(device).float(),
        tr_features=torch.from_numpy(
        tr_features).to(device).float(),
        env_features=torch.from_numpy(
        env_features).to(device).float(),
        node_types=torch.from_numpy(
        node_types).to(device).float(),
        sample_node_length=sample_node_length,
        ev_indexes_node_length=ev_indexes_node_length,
        action_mapper=action_mapper,
        ev_indexes=ev_indexes,
        cs_indexes=cs_indexes,
        tr_indexes=tr_indexes,
        env_indexes=env_indexes)

    # print(f"State batch: {state_batch}")
    # print(f"edge_index shape: {state_batch.edge_index.shape}")
    # print(f"ev_features shape: {state_batch.ev_features.shape}")
    # print(f"cs_features shape: {state_batch.cs_features.shape}")
    # print(f"tr_features shape: {state_batch.tr_features.shape}")
    # print(f"env_features shape: {state_batch.env_features.shape}")
    # print(f"node_types shape: {state_batch.node_types.shape}")
    # print(f"sample_node_length shape: {state_batch.sample_node_length}")
    # print(f"ev_indexes shape: {state_batch.ev_indexes.shape}")
    # print(f"cs_indexes shape: {state_batch.cs_indexes.shape}")
    # print(f"tr_indexes shape: {state_batch.tr_indexes.shape}")

    return state_batch
