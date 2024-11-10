import torch
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, \
    SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import numpy as np

softmax = torch.nn.LogSoftmax(dim=1)


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ResidualGNNs(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.num_layers = num_layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features) / 2) - (num_features / 2) + (hidden_channels * num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels * num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                # print('i=0 xx.shape',xx.shape)# [16000, 1000]
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                # print('i=0 xx.reshaped.shape',xx.shape) # [16, 1000, 1000]
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                # print('i=0 x.shape',x.shape) # 16, 499500
                x = self.bn(x)
            else:
                # print('i>0 xx.shape',xx.shape) # 16000, 32
                xx = self.aggr(xx, batch)
                # print('i>0 xx.shape after aggr',xx.shape) # 16, 32
                h.append(xx)
        if self.num_layers > 0:
            h = torch.cat(h, dim=1)
            # print('h.shape',h.shape,'\n=') # [16, 96]
            h = self.bnh(h)
            # print('h.shape',h.shape,'\n===')
            x = torch.cat((x, h), dim=1)
        # print('x_h.shape',x.shape,'\n=========') # [16, 499596]
        x = self.mlp(x)
        return softmax(x)


class ResidualGNNsWithGNNOnlyCorrInput(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.num_layers = num_layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = hidden_channels * num_layers  # int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels * num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                # print('i=0 xx.shape',xx.shape)# [16000, 1000]
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                # print('i=0 xx.reshaped.shape',xx.shape) # [16, 1000, 1000]
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                # print('i=0 x.shape',x.shape) # 16, 499500
                x = self.bn(x)
            else:
                # print('i>0 xx.shape',xx.shape) # 16000, 32
                xx = self.aggr(xx, batch)
                # print('i>0 xx.shape after aggr',xx.shape) # 16, 32
                h.append(xx)
        if self.num_layers > 0:
            h = torch.cat(h, dim=1)
            # print('h.shape',h.shape,'\n=') # [16, 96]
            h = self.bnh(h)
            # print('h.shape',h.shape,'\n===')
            # x = torch.cat((x,h),dim=1)
        # print('x_h.shape',x.shape,'\n=========') # [16, 499596]
        x = self.mlp(h)
        return softmax(x)


class ResidualGNNsWithGNNOnlyCorrInputWithoutAggr(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        # self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.num_layers = num_layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = hidden_channels * num_layers  # int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bnhs = [nn.BatchNorm1d(hidden_channels * 1000).cuda(),
                     nn.BatchNorm1d(hidden_channels * 1000).cuda(),
                     nn.BatchNorm1d(hidden_channels * 1000).cuda()]
        # self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 1000, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batchsize = x.shape[0] // x.shape[-1]
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]

        for i, xx in enumerate(xs[1:]):
            if i == 0:
                xx = xx.reshape(batchsize, xx.shape[0] * xx.shape[1] // batchsize)
                h = self.bnhs[i](xx)
            else:
                xx = xx.reshape(batchsize, xx.shape[0] * xx.shape[1] // batchsize)
                h += self.bnhs[i](xx)
        # print(h.shape,' xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        # if self.num_layers>0:
        #     # h = self.bnh(h)
        #     # print('h.shape',h.shape,'\n===')
        #     # x = torch.cat((x,h),dim=1)
        # # print('x_h.shape',x.shape,'\n=========') # [16, 499596]
        x = self.mlp(h)
        return softmax(x)


class ResidualGNNsWithoutAggr(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        # self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.num_layers = num_layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = hidden_channels * num_layers  # int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bnhs = [nn.BatchNorm1d(hidden_channels * 1000).cuda(),
                     nn.BatchNorm1d(hidden_channels * 1000).cuda(),
                     nn.BatchNorm1d(hidden_channels * 1000).cuda()]
        self.bn = nn.BatchNorm1d(int((num_features * num_features) / 2 - num_features / 2))
        self.mlp = nn.Sequential(
            nn.Linear(531500, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batchsize = x.shape[0] // x.shape[-1]
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]

        x = x.reshape(data.num_graphs, x.shape[1], -1)
        # print('i=0 xx.reshaped.shape',xx.shape) # [16, 1000, 1000]
        x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in x])
        # print('i=0 x.shape',x.shape) # 16, 499500
        x = self.bn(x)

        for i, xx in enumerate(xs[1:]):
            if i == 0:
                xx = xx.reshape(batchsize, xx.shape[0] * xx.shape[1] // batchsize)
                h = self.bnhs[i](xx)
            else:
                xx = xx.reshape(batchsize, xx.shape[0] * xx.shape[1] // batchsize)
                h += self.bnhs[i](xx)
        # print(h.shape,' xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        # if self.num_layers>0:
        #     # h = self.bnh(h)
        #     # print('h.shape',h.shape,'\n===')
        x = torch.cat((x, h), dim=1)
        # print('x_h.shape',x.shape,'\n=========') # [16, 499596]
        x = self.mlp(x)
        return softmax(x)


class ResidualGNNsWithGNNOnlyNodeIndexInputWithoutAggr(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        # self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.num_layers = num_layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = hidden_channels * num_layers  # int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bnhs = [nn.BatchNorm1d(hidden_channels * 1000).cuda(),
                     nn.BatchNorm1d(hidden_channels * 1000).cuda(),
                     nn.BatchNorm1d(hidden_channels * 1000).cuda()]
        # self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 1000, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batchsize = x.shape[0] // x.shape[-1]
        identity_matrix = torch.eye(1000)
        x = identity_matrix.unsqueeze(0).repeat(batchsize, 1, 1)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).cuda()
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]

        for i, xx in enumerate(xs[1:]):
            if i == 0:
                xx = xx.reshape(batchsize, xx.shape[0] * xx.shape[1] // batchsize)
                h = self.bnhs[i](xx)
            else:
                xx = xx.reshape(batchsize, xx.shape[0] * xx.shape[1] // batchsize)
                h += self.bnhs[i](xx)
        # print(h.shape,' xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        # if self.num_layers>0:
        #     # h = self.bnh(h)
        #     # print('h.shape',h.shape,'\n===')
        #     # x = torch.cat((x,h),dim=1)
        # # print('x_h.shape',x.shape,'\n=========') # [16, 499596]
        x = self.mlp(h)
        return softmax(x)


class Model3(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.num_layers = num_layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features) / 2) - (num_features / 2) + (hidden_channels * num_layers))
        input_dim1 = 96  # int(hidden_channels*num_layers)
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(input_dim1)  # (hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(499500, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batchsize = x.shape[0] // x.shape[-1]
        rawx = x
        identity_matrix = torch.eye(1000)
        x = identity_matrix.unsqueeze(0).repeat(batchsize, 1, 1)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).cuda()
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []

        for i, xx in enumerate(xs):
            if i == 0:
                pass
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        emb = torch.cat(h, dim=1)
        h = self.bnh(emb)
        rawx = rawx.reshape(data.num_graphs, rawx.shape[1], -1)
        # print(rawx.shape,'rawx.shape .....')
        x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in rawx])
        # print('i=0 x.shape',x.shape) # 16, 499500
        x = self.bn(x)
        # print(x.shape,'x.shape .....')
        h = torch.cat([x, h], dim=1)
        x = self.mlp(x)

        return softmax(x)


########################################################## other model

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.datasets import TUDataset
from torch.nn import Linear
from torch.nn import Linear, BatchNorm1d


class GCN(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.SoftmaxAggregation(learn=True)  # aggr.MeanAggregation()
        # self.global_pool = aggr.SortAggregation(k=1)
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.num_layers = num_layers
        if False:
            pass
        else:
            if num_layers > 0:
                self.convs.append(GCNConv(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features) / 2) - (num_features / 2) + (hidden_channels * num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels * num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batchsize = x.shape[0] // x.shape[-1]

        for conv in self.convs:
            x = conv(x, edge_index).relu()

        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return softmax(x)