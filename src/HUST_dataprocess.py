import pandas as pd
import re
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from utility.data_func import CancerTypeDic

class HUSTData:
    def __init__(self):
        self.data = pd.read_csv('/RNAData/HUST/data.csv')
        self.data.index = self.data.loc[:, "sampleId"]
        self.data.drop(columns="sampleId", inplace=True)

        self.labels = set()
        self.cancer_type_dic = CancerTypeDic()
        self.odata = self.data.copy()

        for index in self.data.index:
            cate = self.search_cate(self.data.loc[index, "Organ"], "Hust")
            if cate != -1:
                self.data.loc[index, "Organ"] = cate
                self.labels.add(self.cancer_type_dic.IDtoType(cate, "Hust"))
            else:
                self.data.drop(labels=index, inplace=True)

        self.data.rename(columns={'sampleId': 'Id', 'Organ': 'Label'}, inplace=True)

        # drop part of the Healthy Control
        # self.drop_list = []
        # self.drop_nodes = set()
        # node_id = 0
        # for index, row in self.data.iterrows():
        #     if row['Label'] == 0 and np.random.rand() < 0.6:
        #         self.drop_list.append(index)
        #         self.drop_nodes.add(node_id)
        #     node_id += 1
        # self.data.drop(index=self.drop_list, inplace=True)

    def generate(self):
        df_train, df_test = train_test_split(self.data, test_size=0.4, stratify=self.data['Label'])
        df_test, df_val = train_test_split(self.data, test_size=0.5, stratify=self.data['Label'])

        return df_train, df_test, df_val

    def search_cate(self, str, dict_name):
        if re.search('HD', str) is not None or re.search('Control', str) is not None:
            return self.cancer_type_dic.TypetoID("Health", dict_name)
        elif re.search('Breast', str) is not None or re.search('BrCa', str) is not None:
            return self.cancer_type_dic.TypetoID("Breast", dict_name)
        elif re.search('Thyroid', str) is not None:
            return self.cancer_type_dic.TypetoID("Thyroid", dict_name)
        elif re.search('Urinary', str) is not None:
            return self.cancer_type_dic.TypetoID("Urinary", dict_name)
        elif re.search('Stomach', str) is not None:
            return self.cancer_type_dic.TypetoID("Stomach", dict_name)
        elif re.search('ColonRectum', str) is not None:
            return self.cancer_type_dic.TypetoID("CRC", dict_name)
        elif re.search('Lung', str) is not None:
            return self.cancer_type_dic.TypetoID("Lung", dict_name)
        elif re.search('LiverAndPancreatic', str) is not None:
            return self.cancer_type_dic.TypetoID("LiverAndPancreatic", dict_name)
        else:
            return self.cancer_type_dic.TypetoID("Unknown", dict_name)

    def plot(self):
        x_list = list(self.labels)
        y_list = [0]*len(self.labels)
        for row_id, row in self.data.iterrows():
            label = row['Label']
            y_list[x_list.index(self.cancer_type_dic.IDtoType(label, "Hust"))] += 1
        # plot
        fig, ax = plt.subplots()
        ax.bar(x_list, y_list, width=1, edgecolor="white", linewidth=0.7)
        plt.show()

    def recover(self):
        recoverdata = self.data.copy()
        recoverdata = recoverdata - recoverdata.min()["ENSG00000227232"]
        recoverdata.T.to_csv("Hust.csv")

class NewHustData:
    def __init__(self):
        self.data = pd.read_csv('RNAData/HUST/DroppedData.csv')
        self.data.index = self.data.loc[:, "sampleId"]
        self.data.drop(columns="sampleId", inplace=True)

if __name__ == '__main__':
    hust_data = NewHustData()
    df = hust_data.data.drop(columns="Label")
    df = df.iloc[:, :1000]
    pearson = df.corr(method="pearson")
    import seaborn as sns
    import matplotlib.pyplot as plt
    # 随机生成数据
    print(pearson)
    plt.rcParams['axes.unicode_minus'] = False
    sns.heatmap(pearson, linewidths=0.1, square=True, linecolor='white', annot=True, cmap="Reds")
    plt.title('Pearson')
    plt.savefig("pearson.png")
    print(pearson)
    # df_train, df_test, df_val = hust_data.generate()
    # hust_data.plot()