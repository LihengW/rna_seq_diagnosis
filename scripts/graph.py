import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Dataset.GSE.TEP_dataprocess import GSEData
from Dataset.HUST.HUST_dataprocess import HUSTData

class Colorlist:
    def __init__(self):
        self.colors = ['#FFDAB9', '#F0FFFF', '#FFE4E1', '#191970', '#6A5ACD', '#00EEEE',
                       '#C1FFC1', '#43CD80', '#ADD8E6', '#FFF68F', '#FFD700', '#FF8247']
        self.hust_colors = {
            'HD': '#FFDAB9', 'Stomach': '#F0FFFF', 'ColonRectum': '#FFE4E1', 'Lung': '#191970',
            'Thyroid': '#6A5ACD', 'Urinary': '#00EEEE', 'LiverAndPancreatic': '#C1FFC1', 'Breast': '#C1FFC1'
        }
    def getcolor_gse(self, num):
        return self.colors[num % len(self.colors)]

    def getcolor_hust(self, organ):
        return self.hust_colors[organ]
def generate_graph_gse(filepath):
    G = nx.Graph()
    graph_data = pd.read_csv(filepath)
    nodes_data = GSEData()
    colorlist = Colorlist()
    for row in graph_data.iterrows():
        G.add_edge(row[1][0], row[1][1], weight=row[1][2])
    for n in G:
        n = int(n)
        cancer = nodes_data.data.iloc[n]['Label']
        G.nodes[n]['cancer_type'] = cancer
        G.nodes[n]['color'] = colorlist.getcolor_gse(cancer)
        print(nodes_data.data.iloc[n]['Label'])
        G.nodes[n]['pos'] = np.random.rand(2)
    pos = nx.get_node_attributes(G, "pos")

    # plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, pos=nx.spectral_layout(G, 'weight'), node_color=list(nx.get_node_attributes(G, "color")), node_size=100,
                     alpha=0.6, with_labels = False, style='dashed')
    # nx.draw_networkx_edges(G, pos, alpha=0.4)
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     node_size=80,
    #     node_color=list(nx.get_node_attributes(G, "color")),
    # )

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.show()

def generate_graph_hust(filepath):
    G = nx.Graph()
    graph_data = pd.read_csv(filepath)
    nodes_data = HUSTData()
    colorlist = Colorlist()
    for row in graph_data.iterrows():
        G.add_edge(row[1][0], row[1][1], weight=row[1][2])
    for n in G:
        n = int(n)
        cancer = nodes_data.odata.iloc[n]['Organ']
        G.nodes[n]['cancer_type'] = cancer
        G.nodes[n]['color'] = colorlist.getcolor_hust(cancer)
        G.nodes[n]['pos'] = np.random.rand(2)
    pos = nx.get_node_attributes(G, "pos")

    # plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, pos=nx.spectral_layout(G, weight='weight'), node_color=list(nx.get_node_attributes(G, "color")), edgelist=[],
                     alpha=0.8, with_labels=False, style='dashed')
    # nx.draw_networkx_edges(G, pos, alpha=0.4)
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     node_size=80,
    #     node_color=list(nx.get_node_attributes(G, "color")),
    # )

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.show()



generate_graph_hust("assets\HUST_graph.csv")