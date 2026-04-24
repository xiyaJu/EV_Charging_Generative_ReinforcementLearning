import torch
import torch.nn as nn

from DT.models.graph_components import GraphStateEncoder


def _scatter_node_actions(batch_graph, node_action_values, batch_size, act_dim, device, grad_anchor=None):
    sample_actions = []
    cursor = 0
    for num_evs in batch_graph.ev_indexes_node_length:
        sample_action = torch.zeros((act_dim,), device=device, dtype=torch.float32)
        if num_evs > 0:
            mapper = batch_graph.action_mapper[cursor: cursor + num_evs]
            sample_action = sample_action.scatter(
                0, mapper, node_action_values[cursor: cursor + num_evs]
            )
        sample_actions.append(sample_action)
        cursor += num_evs
    action_preds = torch.stack(sample_actions, dim=0) if sample_actions else torch.zeros(
        (batch_size, act_dim), device=device, dtype=torch.float32
    )
    if grad_anchor is not None:
        action_preds = action_preds + 0.0 * grad_anchor
    return torch.tanh(action_preds)


class NodewiseGraphActor(nn.Module):
    def __init__(self, act_dim, fx_node_sizes, feature_dim=8, hidden_size=128, gnn_hidden_dim=32, num_gcn_layers=3):
        super().__init__()
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.encoder = GraphStateEncoder(
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.ev_action_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, graph_states, config=None):
        batch_size = len(graph_states)
        pooled, node_embeddings, batch_graph = self.encoder(
            graph_states,
            config=config,
            batch_size=batch_size,
            seq_length=1,
        )
        context = self.context_proj(pooled[:, 0, :])

        node_action_values = []
        cursor = 0
        for sample_idx, num_evs in enumerate(batch_graph.ev_indexes_node_length):
            if num_evs > 0:
                ev_indexes = batch_graph.ev_indexes[cursor: cursor + num_evs]
                ev_embeddings = node_embeddings[ev_indexes]
                repeated_context = context[sample_idx].unsqueeze(0).expand(num_evs, -1)
                action_values = self.ev_action_head(
                    torch.cat([ev_embeddings, repeated_context], dim=-1)
                ).squeeze(-1)
                node_action_values.append(action_values)
            cursor += num_evs

        if node_action_values:
            node_action_values = torch.cat(node_action_values, dim=0)
        else:
            node_action_values = torch.empty((0,), device=context.device, dtype=torch.float32)

        return _scatter_node_actions(
            batch_graph=batch_graph,
            node_action_values=node_action_values,
            batch_size=batch_size,
            act_dim=self.act_dim,
            device=context.device,
            grad_anchor=context.sum(),
        )


class GraphQNetwork(nn.Module):
    def __init__(self, act_dim, fx_node_sizes, feature_dim=8, hidden_size=128, gnn_hidden_dim=32, num_gcn_layers=3):
        super().__init__()
        self.encoder = GraphStateEncoder(
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, graph_states, actions, config=None):
        batch_size = actions.shape[0]
        pooled, _, _ = self.encoder(graph_states, config=config, batch_size=batch_size, seq_length=1)
        state_embedding = pooled[:, 0, :]
        return self.q_head(torch.cat([state_embedding, actions], dim=-1)).squeeze(-1)


class GraphValueNetwork(nn.Module):
    def __init__(self, fx_node_sizes, feature_dim=8, hidden_size=128, gnn_hidden_dim=32, num_gcn_layers=3):
        super().__init__()
        self.encoder = GraphStateEncoder(
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        self.v_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, graph_states, config=None):
        batch_size = len(graph_states)
        pooled, _, _ = self.encoder(graph_states, config=config, batch_size=batch_size, seq_length=1)
        return self.v_head(pooled[:, 0, :]).squeeze(-1)


class GraphIQLPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        fx_node_sizes,
        feature_dim=8,
        hidden_size=128,
        gnn_hidden_dim=32,
        num_gcn_layers=3,
        device=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device
        self.actor = NodewiseGraphActor(
            act_dim=act_dim,
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        self.q1 = GraphQNetwork(
            act_dim=act_dim,
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        self.q2 = GraphQNetwork(
            act_dim=act_dim,
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        self.value = GraphValueNetwork(
            fx_node_sizes=fx_node_sizes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )

    def act(self, graph_states, config=None):
        return self.actor(graph_states, config=config)

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go=None,
        timesteps=None,
        action_mask=None,
        graph_states=None,
        config=None,
        **kwargs,
    ):
        if graph_states is None:
            raise ValueError("GraphIQLPolicy.get_action requires graph_states for adaptive graph inference.")
        latest_graph_state = [graph_states[-1]]
        action = self.act(latest_graph_state, config=config)[0]
        if action_mask is not None:
            latest_mask = action_mask.reshape(-1, self.act_dim)[-1]
            action = action * latest_mask.to(dtype=action.dtype, device=action.device)
        return action
