import torch
import torch_geometric.data
import pandas as pd
import numpy as np

class GraphData_Hust:
    def __init__(self):
        from HUST_dataprocess import NewHustData
        self.hust_data = NewHustData()
        x = self.hust_data.data
        labels = x.loc[:, "Label"].array
        self.num_classes = len(set(labels))
        y = np.zeros((len(labels), self.num_classes))
        for i in range(len(labels)):
            y[i][labels[i]] = 1
        y = torch.tensor(y).float()
        x.drop("Label", axis=1, inplace=True)
        x = torch.tensor(x.to_numpy().astype(float)).float()

        try:
            edge_index = pd.read_csv("graph/NewHust_graph.csv")
        except:
            edge_index = pd.read_csv("graph/NewHust_graph.csv")

        # drop part of the healthy Control
        # drop_list = []
        # for index, row in edge_index.iterrows():
        #     if row['NodeA'] in self.hust_data.drop_nodes or row['NodeB'] in self.hust_data.drop_nodes:
        #         drop_list.append(index)
        # edge_index.drop(index=drop_list, inplace=True)

        edge_index = torch.tensor(edge_index.loc[(edge_index.Weights > 0), :].iloc[:, 0:2].T.to_numpy())

        self.data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)

        # PyTorch tensor functionality:
        # self.data = self.data.pin_memory()
        # self.data = self.data.to('cuda:0', non_blocking=True)

        self.num_node, self.num_features = x.shape[0], x.shape[1]
        self.num_classes = len(set(NewHustData().data.loc[:, "Label"].array))

        train_mask = [False]*self.num_node
        test_mask = [False]*self.num_node
        for i in range(self.num_node):
            if np.random.rand() > 0.7:
                test_mask[i] = True
            else:
                train_mask[i] = True

        self.data.train_mask = torch.tensor(train_mask)
        self.data.test_mask = torch.tensor(test_mask)


class GraphData_GSE:
    def __init__(self):
        from TEP_dataprocess import GSEData
        self.gse_data = GSEData()
        x = self.gse_data.data
        labels = x.loc[:, "Label"].array
        self.num_classes = len(set(labels))
        y = np.zeros((len(labels), self.num_classes))
        for i in range(len(labels)):
            y[i][labels[i]] = 1
        y = torch.tensor(y).float()
        x.drop("Label", axis=1, inplace=True)
        x = torch.tensor(x.to_numpy().astype(float)).float()

        try:
            edge_index = pd.read_csv("graph/GSE_graph.csv")
        except:
            edge_index = pd.read_csv("graph/GSE_graph.csv")

        # drop part of the healthy Control
        # drop_list = []
        # for index, row in edge_index.iterrows():
        #     if row['NodeA'] in self.hust_data.drop_nodes or row['NodeB'] in self.hust_data.drop_nodes:
        #         drop_list.append(index)
        # edge_index.drop(index=drop_list, inplace=True)

        edge_index = torch.tensor(edge_index.loc[(edge_index.Weights > 0), :].iloc[:, 0:2].T.to_numpy())

        self.data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)

        # PyTorch tensor functionality:
        # self.data = self.data.pin_memory()
        # self.data = self.data.to('cuda:0', non_blocking=True)

        self.num_node, self.num_features = x.shape[0], x.shape[1]

        train_mask = [False]*self.num_node
        test_mask = [False]*self.num_node
        for i in range(self.num_node):
            if np.random.rand() > 0.7:
                test_mask[i] = True
            else:
                train_mask[i] = True

        self.data.train_mask = torch.tensor(train_mask)
        self.data.test_mask = torch.tensor(test_mask)

if __name__ == '__main__':
    t = GraphData_GSE()
    print(t)


