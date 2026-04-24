import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from utils import (
    PST_V2G_ProfitMax_state_to_GNN,
    graph_dict_to_data,
)


def make_empty_graph_state():
    return Data(
        ev_features=np.empty((0, 5), dtype=np.float32),
        cs_features=np.empty((0, 4), dtype=np.float32),
        tr_features=np.empty((0, 2), dtype=np.float32),
        env_features=np.zeros((1, 6), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        node_types=np.array([0], dtype=np.int64),
        sample_node_length=[1],
        action_mapper=np.empty((0,), dtype=np.int64),
        ev_indexes=np.empty((0,), dtype=np.int64),
        cs_indexes=np.empty((0,), dtype=np.int64),
        tr_indexes=np.empty((0,), dtype=np.int64),
        env_indexes=np.array([0], dtype=np.int64),
    )


def _ensure_graph_data(graph_state, config=None):
    if graph_state is None:
        return make_empty_graph_state()
    if isinstance(graph_state, Data):
        return graph_state
    if isinstance(graph_state, dict):
        return graph_dict_to_data(graph_state)
    if isinstance(graph_state, np.ndarray):
        if config is None:
            raise ValueError("config is required when graph_state is a flat numpy state.")
        return PST_V2G_ProfitMax_state_to_GNN(graph_state, config)
    raise TypeError(f"Unsupported graph_state type: {type(graph_state).__name__}")


def normalize_graph_state_batch(graph_states, batch_size=None, seq_length=None):
    if graph_states is None:
        return None
    if batch_size is not None and seq_length is not None and len(graph_states) == batch_size * seq_length:
        nested = []
        offset = 0
        for _ in range(batch_size):
            nested.append(graph_states[offset: offset + seq_length])
            offset += seq_length
        return nested
    if len(graph_states) == 0:
        return []
    if isinstance(graph_states[0], (list, tuple)):
        return [list(row) for row in graph_states]
    return [list(graph_states)]


def to_gnn_batch(state_batch, device=None, config=None):
    nested_states = normalize_graph_state_batch(state_batch)
    if nested_states is None:
        raise ValueError("state_batch cannot be None")

    states = []
    for sublist in nested_states:
        for state in sublist:
            states.append(_ensure_graph_data(state, config=config))

    if len(states) == 0:
        states = [make_empty_graph_state()]

    ev_features_parts = []
    cs_features_parts = []
    tr_features_parts = []
    env_features_parts = []
    node_types_parts = []
    edge_index_parts = []
    action_mapper_parts = []

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = []
    sample_node_length = []
    ev_indexes_node_length = []

    node_offset = 0
    for state in states:
        sample_nodes = int(len(state.node_types))
        sample_node_length.append(sample_nodes)
        ev_indexes_node_length.append(int(len(state.ev_indexes)))

        ev_features_parts.append(np.asarray(state.ev_features, dtype=np.float32).reshape(-1, 5))
        cs_features_parts.append(np.asarray(state.cs_features, dtype=np.float32).reshape(-1, 4))
        tr_features_parts.append(np.asarray(state.tr_features, dtype=np.float32).reshape(-1, 2))
        env_features_parts.append(np.asarray(state.env_features, dtype=np.float32).reshape(-1, 6))
        node_types_parts.append(np.asarray(state.node_types, dtype=np.int64).reshape(-1))

        local_edge_index = np.asarray(state.edge_index, dtype=np.int64).reshape(2, -1)
        if local_edge_index.shape[1] > 0:
            edge_index_parts.append(local_edge_index + node_offset)

        ev_indexes.append(np.asarray(state.ev_indexes, dtype=np.int64).reshape(-1) + node_offset)
        cs_indexes.append(np.asarray(state.cs_indexes, dtype=np.int64).reshape(-1) + node_offset)
        tr_indexes.append(np.asarray(state.tr_indexes, dtype=np.int64).reshape(-1) + node_offset)
        env_indexes.append(np.asarray(state.env_indexes, dtype=np.int64).reshape(-1) + node_offset)

        local_action_mapper = np.asarray(state.action_mapper, dtype=np.int64).reshape(-1)
        action_mapper_parts.append(local_action_mapper)

        node_offset += sample_nodes

    edge_index = np.concatenate(edge_index_parts, axis=1) if edge_index_parts else np.empty((2, 0), dtype=np.int64)
    ev_features = np.concatenate(ev_features_parts, axis=0) if ev_features_parts else np.empty((0, 5), dtype=np.float32)
    cs_features = np.concatenate(cs_features_parts, axis=0) if cs_features_parts else np.empty((0, 4), dtype=np.float32)
    tr_features = np.concatenate(tr_features_parts, axis=0) if tr_features_parts else np.empty((0, 2), dtype=np.float32)
    env_features = np.concatenate(env_features_parts, axis=0) if env_features_parts else np.empty((0, 6), dtype=np.float32)
    node_types = np.concatenate(node_types_parts, axis=0) if node_types_parts else np.empty((0,), dtype=np.int64)
    action_mapper = np.concatenate(action_mapper_parts, axis=0) if action_mapper_parts else np.empty((0,), dtype=np.int64)
    ev_indexes_arr = np.concatenate(ev_indexes, axis=0) if ev_indexes else np.empty((0,), dtype=np.int64)
    cs_indexes_arr = np.concatenate(cs_indexes, axis=0) if cs_indexes else np.empty((0,), dtype=np.int64)
    tr_indexes_arr = np.concatenate(tr_indexes, axis=0) if tr_indexes else np.empty((0,), dtype=np.int64)
    env_indexes_arr = np.concatenate(env_indexes, axis=0) if env_indexes else np.empty((0,), dtype=np.int64)

    return Data(
        edge_index=torch.from_numpy(edge_index).to(device=device, dtype=torch.long),
        ev_features=torch.from_numpy(ev_features).to(device=device, dtype=torch.float32),
        cs_features=torch.from_numpy(cs_features).to(device=device, dtype=torch.float32),
        tr_features=torch.from_numpy(tr_features).to(device=device, dtype=torch.float32),
        env_features=torch.from_numpy(env_features).to(device=device, dtype=torch.float32),
        node_types=torch.from_numpy(node_types).to(device=device, dtype=torch.long),
        sample_node_length=sample_node_length,
        ev_indexes_node_length=ev_indexes_node_length,
        action_mapper=torch.from_numpy(action_mapper).to(device=device, dtype=torch.long),
        ev_indexes=torch.from_numpy(ev_indexes_arr).to(device=device, dtype=torch.long),
        cs_indexes=torch.from_numpy(cs_indexes_arr).to(device=device, dtype=torch.long),
        tr_indexes=torch.from_numpy(tr_indexes_arr).to(device=device, dtype=torch.long),
        env_indexes=torch.from_numpy(env_indexes_arr).to(device=device, dtype=torch.long),
    )


class GraphStateEncoder(nn.Module):
    def __init__(
        self,
        fx_node_sizes,
        feature_dim=8,
        hidden_size=128,
        gnn_hidden_dim=32,
        num_gcn_layers=3,
    ):
        super().__init__()
        self.fx_node_sizes = fx_node_sizes
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gcn_layers = num_gcn_layers

        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        self.gcn_conv = GCNConv(feature_dim, gnn_hidden_dim)
        if num_gcn_layers == 3:
            self.gcn_layers = nn.ModuleList(
                [GCNConv(gnn_hidden_dim, 2 * gnn_hidden_dim), GCNConv(2 * gnn_hidden_dim, hidden_size)]
            )
        elif num_gcn_layers == 4:
            self.gcn_layers = nn.ModuleList(
                [
                    GCNConv(gnn_hidden_dim, 2 * gnn_hidden_dim),
                    GCNConv(2 * gnn_hidden_dim, 3 * gnn_hidden_dim),
                    GCNConv(3 * gnn_hidden_dim, hidden_size),
                ]
            )
        elif num_gcn_layers == 5:
            self.gcn_layers = nn.ModuleList(
                [
                    GCNConv(gnn_hidden_dim, 2 * gnn_hidden_dim),
                    GCNConv(2 * gnn_hidden_dim, 3 * gnn_hidden_dim),
                    GCNConv(3 * gnn_hidden_dim, 4 * gnn_hidden_dim),
                    GCNConv(4 * gnn_hidden_dim, hidden_size),
                ]
            )
        else:
            raise ValueError("num_gcn_layers must be one of {3, 4, 5}")

    def forward(self, graph_states, config=None, batch_size=None, seq_length=None):
        nested_graph_states = normalize_graph_state_batch(graph_states, batch_size=batch_size, seq_length=seq_length)
        if nested_graph_states is None:
            raise ValueError("graph_states cannot be None")
        if batch_size is None:
            batch_size = len(nested_graph_states)
        if seq_length is None:
            seq_length = len(nested_graph_states[0]) if nested_graph_states else 1

        graph_batch = to_gnn_batch(nested_graph_states, device=next(self.parameters()).device, config=config)

        total_nodes = (
            graph_batch.ev_features.shape[0]
            + graph_batch.cs_features.shape[0]
            + graph_batch.tr_features.shape[0]
            + graph_batch.env_features.shape[0]
        )
        embedded_x = torch.zeros(total_nodes, self.feature_dim, device=next(self.parameters()).device, dtype=torch.float32)

        if graph_batch.ev_indexes.numel() > 0:
            embedded_x[graph_batch.ev_indexes] = self.ev_embedding(graph_batch.ev_features)
        if graph_batch.cs_indexes.numel() > 0:
            embedded_x[graph_batch.cs_indexes] = self.cs_embedding(graph_batch.cs_features)
        if graph_batch.tr_indexes.numel() > 0:
            embedded_x[graph_batch.tr_indexes] = self.tr_embedding(graph_batch.tr_features)
        if graph_batch.env_indexes.numel() > 0:
            embedded_x[graph_batch.env_indexes] = self.env_embedding(graph_batch.env_features)

        x_gnn = F.relu(self.gcn_conv(F.relu(embedded_x), graph_batch.edge_index))
        for layer in self.gcn_layers:
            x_gnn = layer(x_gnn, graph_batch.edge_index)
            if layer is not self.gcn_layers[-1]:
                x_gnn = F.relu(x_gnn)

        pool_batch = np.repeat(np.arange(len(graph_batch.sample_node_length)), graph_batch.sample_node_length)
        pool_batch = torch.from_numpy(pool_batch).to(device=next(self.parameters()).device, dtype=torch.long)
        pooled_embedding = global_mean_pool(x_gnn, batch=pool_batch)
        pooled_embedding = pooled_embedding.reshape(batch_size, seq_length, -1)

        return pooled_embedding, x_gnn, graph_batch
