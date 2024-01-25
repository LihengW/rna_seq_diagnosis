import torch
import torch_geometric.data
import pandas as pd
import numpy as np
from utility.data_func import person_graph_build


class PublicGeneList:
    def __init__(self):
        from HUST_dataprocess import NewHustData
        hust_data = NewHustData()
        from gse_dataprocess import GSE183635Data
        gse_data2 = GSE183635Data()

        gse_data2_gene = set(gse_data2.data.columns[1:])
        hust_data_gene = set(hust_data.data.columns[1:])

        self.public_gene = gse_data2_gene.intersection(hust_data_gene)

    def get(self):
        return list(self.public_gene)


class GraphData_GSE183635:
    def __init__(self, normalized=True, pearson_graph=False, gene_filter=None, person_index=0.5, cate_filter_num=50):
        from gse_dataprocess import GSE183635Data
        self.gse_data = GSE183635Data()
        self.gse_data.filter(cate_filter_num)

        x = self.gse_data.data

        # remap train labels
        self.train_label_map = {}
        remap_index = 0

        labels = x.loc[:, "Label"].array
        self.labels = labels
        self.num_classes = len(set(labels))
        self.class_weight = [0]*self.num_classes
        for i in labels:
            if i not in self.train_label_map:
                self.train_label_map[i] = remap_index
                remap_index += 1
            self.class_weight[self.train_label_map[i]] += 1

        for j in range(self.num_classes):
            self.class_weight[j] = 1 / self.class_weight[j] * 1000

        y = np.zeros((len(labels), self.num_classes))

        for i in range(len(labels)):
            y[i][self.train_label_map[labels[i]]] = 1
        y = torch.tensor(y).float()
        x.drop("Label", axis=1, inplace=True)

        #gene_filter
        if gene_filter:
            x = x.loc[:, gene_filter]

        if normalized:
            for col in x.columns:
                max = x[col].max()
                if max:
                    x[col] = x[col] / max

        if not pearson_graph:
            try:
                edge_index = pd.read_csv("graph/GSE183635_graph.csv")
            except:
                edge_index = pd.read_csv("graph/GSE183635_graph.csv")

            edge_index = torch.tensor(edge_index.loc[(edge_index.Weights > 0), :].iloc[:, 0:2].T.to_numpy())
        else:
            edge_index = person_graph_build(x.T, person_index)
            edge_index = torch.tensor(edge_index)

        x = torch.tensor(x.to_numpy().astype(float)).float()


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



class GraphData_Hust:
    def __init__(self, normalized=True, pearson_graph=False, gene_filter=None, person_index=0.5):
        from HUST_dataprocess import NewHustData
        self.hust_data = NewHustData()
        x = self.hust_data.data

        labels = x.loc[:, "Label"].array
        self.labels = labels
        self.num_classes = len(set(labels))
        self.class_weight = [0]*self.num_classes
        for i in labels:
            self.class_weight[i] += 1
        for j in range(self.num_classes):
            self.class_weight[j] = 1 / self.class_weight[j] * 1000

        y = np.zeros((len(labels), self.num_classes))
        for i in range(len(labels)):
            y[i][labels[i]] = 1
        y = torch.tensor(y).float()
        x.drop("Label", axis=1, inplace=True)

        if gene_filter:
            x = x.loc[:, gene_filter]

        if normalized:
            for col in x.columns:
                max = x[col].max()
                x[col] = x[col] / max

        if not pearson_graph:
            try:
                edge_index = pd.read_csv("graph/NewHust_graph.csv")
            except:
                edge_index = pd.read_csv("graph/NewHust_graph.csv")

            edge_index = torch.tensor(edge_index.loc[(edge_index.Weights > 0), :].iloc[:, 0:2].T.to_numpy())
        else:
            edge_index = person_graph_build(x.T, person_index)
            edge_index = torch.tensor(edge_index)

        x = torch.tensor(x.to_numpy().astype(float)).float()

        self.data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)

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


class GraphData_GSE68086:
    def __init__(self, normalized=True, pearson_graph=False, gene_filter=None, person_index=0.5):
        from TEP_dataprocess import GSEData
        self.gse_data = GSEData()
        x = self.gse_data.data

        labels = x.loc[:, "Label"].array
        self.labels = labels
        self.num_classes = len(set(labels))
        self.class_weight = [0]*self.num_classes
        for i in labels:
            self.class_weight[i] += 1
        for j in range(self.num_classes):
            self.class_weight[j] = 1 / self.class_weight[j] * 1000

        y = np.zeros((len(labels), self.num_classes))
        for i in range(len(labels)):
            y[i][labels[i]] = 1
        y = torch.tensor(y).float()
        x.drop("Label", axis=1, inplace=True)

        if gene_filter:
            x = x.loc[:, gene_filter]

        if normalized:
            for col in x.columns:
                max = x[col].max()
                if max:
                    x[col] = x[col] / max

        if not pearson_graph:
            try:
                edge_index = pd.read_csv("graph/GSE_graph.csv")
            except:
                edge_index = pd.read_csv("graph/GSE_graph.csv")

            edge_index = torch.tensor(edge_index.loc[(edge_index.Weights > 0), :].iloc[:, 0:2].T.to_numpy())
        else:
            edge_index = person_graph_build(x.T, person_index)
            edge_index = torch.tensor(edge_index)

        x = torch.tensor(x.to_numpy().astype(float)).float()

        self.data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)

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
    public_gene = PublicGeneList().get()
    g = GraphData_GSE183635(pearson_graph=True, gene_filter=public_gene)
    print(g.data.edge_index)
    g2 = GraphData_Hust()
    print(g2.data.edge_index)
