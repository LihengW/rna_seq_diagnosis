# torch library
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.nn
# original library
import dataset

# Default GAT Module
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 32
        self.in_head = 4
        self.out_head = 1

        data = dataset.GraphData_Hust()

        self.lin1 = torch.nn.Linear(data.num_features, self.hid * 16)
        self.lin2 = torch.nn.Linear(self.hid * 16, self.hid * 8)
        self.lin3 = torch.nn.Linear(self.hid * 8, self.hid * 4)

        self.conv1 = torch_geometric.nn.GATConv(self.hid * 4, self.hid * 2, heads=self.in_head, dropout=0.4)
        self.conv2 = torch_geometric.nn.GATConv(self.hid * self.in_head * 2, self.hid, heads=self.in_head, dropout=0.4)
        self.conv3 = torch_geometric.nn.GATConv(self.hid * self.in_head, data.num_classes, concat=False, heads=self.out_head, dropout=0.4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        # return F.softmax(x, dim=1)
        return x

class GATBackup(torch.nn.Module):
    def __init__(self):
        super(GATBackup, self).__init__()
        self.hid = 64
        self.in_head = 8
        self.out_head = 1

        data = dataset.GraphData_Hust()

        self.lin1 = torch.nn.Linear(data.num_features, self.hid * 8)
        self.lin2 = torch.nn.Linear(self.hid * 8, self.hid * 4)
        self.lin3 = torch.nn.Linear(self.hid * 4, self.hid * 2)
        self.conv1 = torch_geometric.nn.GATConv(self.hid * 2, self.hid, heads=self.in_head, dropout=0.4)
        self.conv2 = torch_geometric.nn.GATConv(self.hid * self.in_head, int(self.hid / 2), heads=self.in_head, dropout=0.4)
        self.conv3 = torch_geometric.nn.GATConv(int(self.hid * self.in_head / 2), data.num_classes, concat=False, heads=self.out_head, dropout=0.4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        # return F.softmax(x, dim=1)
        return x


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        data = dataset.GraphData_Hust()
        self.lin1 = Linear(data.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.lin4 = Linear(hidden_channels, data.num_classes)

        self.hidden_channels = hidden_channels
        self.num_classes = data.num_classes
    def forward(self, data):
        x = data.x
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin4(x)
        return x


# SELF-MADE BLOCK ------------------------------------------------------
# out_features at last presents the number of classification
class RNAModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(RNAModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = torch.nn.Sequential()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        try:
            x = self.layers(x)
            return x
        except:
            x, edge_index = self.layers(x, edge_index)
            return x

class MixedModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super(MixedModel, self).__init__()
        self.in_features = model1.in_features
        self.out_features = model2.out_features
        self.layers = torch.nn.Sequential(*(list(model1.layers.children()) + list(model2.layers.children())))
        print(self.layers)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.layers)):
            try:
                x = self.layers[i](x, edge_index)
            except:
                x = self.layers[i](x)
        return x

class FLATMLP(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, out_features, layernum=3, active_function='relu', dropout=0.3):
        super(FLATMLP, self).__init__()
        # torch.manual_seed(12345)
        self.layers = torch.nn.Sequential()
        if layernum == 1:
            self.layers.add_module("lin", torch.nn.Linear(in_features, out_features))
        else:
            # input Linear Layer
            self.layers.add_module("lin1", torch.nn.Linear(in_features, hidden_channels))
            # middle Layer
            for i in range(layernum - 1):
                # dropout and activation function
                if active_function=='relu':
                    self.layers.add_module("Relu" + str(i+1), torch.nn.ReLU())
                elif active_function=='elu':
                    self.layers.add_module("Elu" + str(i+1), torch.nn.ELU())

                self.layers.add_module("dropout" + str(i + 1), torch.nn.Dropout(p=dropout))

                # linear layer
                if i == layernum - 2:
                    self.layers.add_module("lin" + str(layernum), torch.nn.Linear(hidden_channels, out_features))
                else:
                    self.layers.add_module("lin" + str(i+2), torch.nn.Linear(hidden_channels, hidden_channels))

        self.in_features = in_features
        self.out_features = out_features
        self.active_function = active_function
        self.hidden_channels = hidden_channels

        print(self.layers)
    def forward(self, data):
        x = data.x
        x = self.layers(x)
        return x

class DownSamplingMLP(torch.nn.Module):
    def __init__(self, in_features, top_hidden_channels, shrink_rate, out_features, layernum=3, active_function='relu', dropout=0.3):
        super(DownSamplingMLP, self).__init__()
        # torch.manual_seed(12345)
        self.layers = torch.nn.Sequential()

        if pow(shrink_rate, layernum) > top_hidden_channels:
            # shrink to one feature
            raise ValueError

        if layernum == 1:
            self.layers.add_module("lin", torch.nn.Linear(in_features, out_features))
        else:
            # input Linear Layer
            hidden_channels = top_hidden_channels
            self.layers.add_module("lin1", torch.nn.Linear(in_features, hidden_channels))
            # middle Layer
            for i in range(layernum - 1):
                # dropout and activation function
                if active_function=='relu':
                    self.layers.add_module("Relu" + str(i+1), torch.nn.ReLU())
                elif active_function=='elu':
                    self.layers.add_module("Elu" + str(i+1), torch.nn.ELU())

                self.layers.add_module("dropout" + str(i + 1), torch.nn.Dropout(p=dropout))

                # shrink to downsample
                next_hidden_channels = int(hidden_channels / shrink_rate)
                if i == layernum - 2:
                    self.layers.add_module("lin" + str(layernum), torch.nn.Linear(hidden_channels, out_features))
                else:
                    self.layers.add_module("lin" + str(i+2), torch.nn.Linear(hidden_channels, next_hidden_channels))
                hidden_channels = next_hidden_channels

        self.top_hidden_channels = top_hidden_channels
        self.out_features = out_features
        self.in_features = in_features
        self.active_function = active_function

        print(self.layers)
    def forward(self, data):
        x = data.x
        x = self.layers(x)
        return x

class DownSamplingGATBlock(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 top_hidden_channels, shrink_rate=2, layernum=3,
                in_head=8, out_head=1,
                 active_function='relu', dropout=0.3, attention_dropout=0.3
                 ):
        super(DownSamplingGATBlock, self).__init__()
        self.in_head = in_head
        self.out_head = out_head
        self.layers = torch.nn.Sequential()

        if pow(shrink_rate, layernum) > top_hidden_channels:
            # shrink to one feature
            raise ValueError

        if layernum == 1:
            self.layers.add_module("conv", torch_geometric.nn.GATConv(in_features, out_features, heads=self.out_head, dropout=attention_dropout))
        else:
            # input Linear Layer
            hidden_channels = top_hidden_channels
            self.layers.add_module("conv1", torch_geometric.nn.GATConv(in_features, hidden_channels, heads=self.in_head, dropout=attention_dropout))
            # middle Layer
            for i in range(layernum - 1):
                # dropout and activation function
                if active_function == 'relu':
                    self.layers.add_module("Relu" + str(i+1), torch.nn.ReLU())
                elif active_function == 'elu':
                    self.layers.add_module("Elu" + str(i+1), torch.nn.ELU())

                self.layers.add_module("dropout" + str(i + 1), torch.nn.Dropout(p=dropout))

                # shrink to downsample
                next_hidden_channels = int(hidden_channels / shrink_rate)
                if i == layernum - 2:
                    self.layers.add_module("conv" + str(layernum), torch_geometric.nn.GATConv(hidden_channels * self.in_head, out_features, heads=self.out_head, dropout=attention_dropout))
                else:
                    self.layers.add_module("conv" + str(i+2), torch_geometric.nn.GATConv(hidden_channels * self.in_head, next_hidden_channels, heads=self.in_head, dropout=attention_dropout))
                hidden_channels = next_hidden_channels

        self.top_hidden_channels = top_hidden_channels
        self.out_features = out_features
        self.in_features = in_features
        self.active_function = active_function
        print(self.layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.layers[0](x, edge_index)
        for i in range(int((len(self.layers) - 1) / 3)):
            x = self.layers[3 * i + 1](x)  # active
            x = self.layers[3 * i + 2](x)  # dropout
            x = self.layers[3 * i + 3](x, edge_index)

        return x

def ConnectModule(module1, module2):
    newmodule = RNAModel(module1.in_features, module2.out_features)
    RNAModel.layers = torch.nn.Sequential(*(list(module1.layers.children()) + list(module2.layers.children())))
    return newmodule
