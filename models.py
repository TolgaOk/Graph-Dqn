import torch


class EdgeGraphConv(torch.nn.Module):

    def __init__(self, n_edges, in_size, out_size):
        super().__init__()
        init_weight = torch.rand(n_edges, in_size, out_size)
        self.weight = torch.nn.Parameter(init_weight)
        self.bias = torch.nn.Parameter(torch.zeros(out_size))

    def forward(self, adjacency, feature):
        assert len(feature.shape) == 3, "Feature must be 3 dimensional"
        feature = torch.unsqueeze(feature, dim=1)
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adjacency, support)
        # Summing for all edge types(0th dim is the batch dim)
        output = torch.mean(output, dim=1)
        output = output + self.bias.reshape(1, 1, -1)
        output = torch.relu(output)
        return output


class GraphDqnModel(torch.nn.Module):
    def __init__(self, n_edge, n_node, mapsize, n_act):
        super().__init__()

        self.n_node = n_node

        self.egconv1 = EdgeGraphConv(n_edge, n_node, 64)
        self.egconv2 = EdgeGraphConv(n_edge, 64, 64)
        self.egconv3 = EdgeGraphConv(n_edge, 64, 64)
        self.egconv4 = EdgeGraphConv(n_edge, 64, 64)

        self.kgconv1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.kgconv2 = torch.nn.Conv2d(64, 64, 1, padding=0)
        self.kgconv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.kgconv4 = torch.nn.Conv2d(64, 64, 1, padding=0)

        init_weight = torch.rand(64, 64)
        self.pool_weight = torch.nn.Parameter(init_weight)

        self.fc = torch.nn.Linear(mapsize*mapsize*64, 50)
        self.head = torch.nn.Linear(50, n_act)

    def forward(self, adj, objmap):

        # shape: (B, N, S, S)
        bs, _, height, width = objmap.shape

        kg_features = torch.eye(self.n_node).unsqueeze(0)*torch.ones(bs, 1, 1)
        kg_features = kg_features.to(objmap.device)

        # Graph conv
        kg_features = self.egconv1(adj, kg_features)
        kg_features = self.egconv2(adj, kg_features)

        # Broadcast
        state = self.broadcast(kg_features, objmap)

        # KG conv
        skip_state = self.kgconv1(state) + self.kgconv2(state)
        skip_state = torch.relu(skip_state)

        # Pooling
        kg_features = self.pooling(skip_state, objmap)

        #   Graph conv
        kg_features = self.egconv3(adj, kg_features)
        kg_features = self.egconv4(adj, kg_features)

        #   Broadcast
        state = self.broadcast(kg_features, objmap)

        # KG conv
        state = self.kgconv3(skip_state) + self.kgconv4(state)
        state = torch.relu(state)

        # Dense Layers
        state = state.view(-1, height*width*64)
        state = torch.relu(self.fc(state))
        values = self.head(state)
        return values

    def broadcast(self, feature, objectmap):
        """ objectmap: (B, N, H, W)
            feature: (B, N, F)

            objectmap -> (B, N, 1, H, W)
            feature ->   (B, N, F, 1, 1)

            result = objectmap * feature
            result.sum(1)
        """
        assert len(feature.shape) == 3, ("feature argument must be 3"
                                         " dimensional. (B, N, F): batch"
                                         " size, #nodes, #features")
        assert len(objectmap.shape) == 4, ("objectmap argument must be 4"
                                           "dimensional. (B, N, H, W): "
                                           "batch size, #nodes, height, width")
        state_tensor = (objectmap.unsqueeze(2) *
                        feature.reshape(*feature.shape, 1, 1))
        return state_tensor.sum(1)

    def pooling(self, state, objectmap):
        """ objectmap: (B, N, H, W)
            state:     (B, F, H, W)
            weight:    (F, F')

            objectmap -> (B, N, 1, H, W)
            state ->     (B, 1, F, H, W)

            result = objectmap * state
            result.sum((-2, -1))
        """
        assert len(state.shape) == 4, ("state must be 4 dimensional "
                                       "(B, F, H, W): batch size, #features"
                                       ", heigth, width")
        assert len(objectmap.shape) == 4, ("objectmap argument must be 4"
                                           "dimensional. (B, N, H, W): "
                                           "batch size, #nodes, height, width")
        n_occurrence = objectmap.sum((-2, -1))
        feature = (objectmap.unsqueeze(2) *
                   state.unsqueeze(1))
        feature = torch.matmul(feature.sum((-2, -1)), self.pool_weight)
        return feature / n_occurrence.unsqueeze(-1)


class VanillaDqnModel(torch.nn.Module):

    def __init__(self, n_node, mapsize, n_act):
        super().__init__()
        self.mapsize = mapsize
        self.conv1 = torch.nn.Conv2d(n_node, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = torch.nn.Linear(64*mapsize*mapsize, 64)
        self.head = torch.nn.Linear(64, n_act)

    def forward(self, state):
        state = torch.relu(self.conv1(state))
        state = torch.relu(self.conv2(state))

        state = state.reshape(-1, self.mapsize*self.mapsize*64)
        state = torch.relu(self.fc1(state))
        values = self.head(state)
        return values

    def reset_weights(self):
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.apply(param_init)


class FullyConnectedModel(torch.nn.Module):

    def __init__(self, in_size, n_act):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_size, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        # self.fc3 = torch.nn.Linear(32, 32)
        self.head = torch.nn.Linear(32, n_act)

        self.reset_weights()

    def forward(self, state):
        state = torch.relu(self.fc1(state))
        state = torch.relu(self.fc2(state))
        # state = torch.relu(self.fc3(state))
        values = self.head(state)
        return values

    def reset_weights(self):
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
        self.apply(param_init)

if __name__ == "__main__":
    K = 12
    N = 20
    B = 128
    F = 6
    D = 3
    S = 10
    A = 4

    adj = torch.ones(K, N, N)
    objmap = torch.randint(0, 1, size=(B, N, S, S)).float()

    model = GraphDqnModel(K, N, S, A)
    values = model(adj, objmap)
    print(values.shape)
