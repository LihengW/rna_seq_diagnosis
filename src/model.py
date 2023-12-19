# torch library
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.nn
# original library
import dataset

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


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 32
        self.in_head = 4
        self.out_head = 1

        data = dataset.GraphData_Hust()

        self.lin1 = torch.nn.Linear(data.num_features, self.hid * 8)
        self.lin2 = torch.nn.Linear(self.hid * 8, self.hid * 2)
        self.conv1 = torch_geometric.nn.GATConv(self.hid * 2, self.hid, heads=self.in_head, dropout=0.4)
        self.conv2 = torch_geometric.nn.GATConv(self.hid * self.in_head, int(self.hid / 2), heads=self.in_head, dropout=0.4)
        self.conv3 = torch_geometric.nn.GATConv(int(self.hid * self.in_head / 2), data.num_classes, concat=False, heads=self.out_head, dropout=0.4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
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
