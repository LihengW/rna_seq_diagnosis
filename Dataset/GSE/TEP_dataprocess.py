import pandas as pd
import re
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from Utility.data_func import CancerTypeDic

class GSEData:
    def __init__(self):
        self.data = pd.read_csv('F:/RNA_seq/Dataset/GSE/GSE68086_TEP_data_matrix.csv')
        self.data = self.data.T
        self.data.columns = self.data.loc["sampleId"]
        self.data.drop(labels="sampleId", inplace=True)

        self.cancer_type_dic = CancerTypeDic()
        for index in self.data.index:
            cate = self.fill_in_cates(index)
            if cate != -1:
                self.data.loc[index]["Organ"] = cate
            else:
                self.data.drop(labels=index, inplace=True)

        self.data.rename(columns={'sampleId': 'Id', 'Organ': 'Label'}, inplace=True)

    def generate(self):
        df_train, df_test = train_test_split(self.data, test_size=0.4, stratify=self.data['Label'])
        df_test, df_val = train_test_split(self.data, test_size=0.5, stratify=self.data['Label'])
        df_test.drop(columns='Label', inplace=True)

        return df_train, df_test, df_val

    def fill_in_cates(self, str):
        if re.search('HD', str) is not None or re.search('Control', str) is not None:
            return self.cancer_type_dic.TypetoID("Health")
        elif re.search('Breast', str) is not None or re.search('BrCa', str) is not None:
            return self.cancer_type_dic.TypetoID("Breast")
        elif re.search('Liver', str) is not None:
            return self.cancer_type_dic.TypetoID("Liver")
        elif re.search('Chol', str) is not None:
            return self.cancer_type_dic.TypetoID("Chol")
        elif re.search('CRC', str) is not None:
            return self.cancer_type_dic.TypetoID("CRC")
        elif re.search('GBM', str) is not None:
            return self.cancer_type_dic.TypetoID("GBM")
        elif re.search('Platelet', str) is not None:
            return self.cancer_type_dic.TypetoID("Platelet")
        elif re.search('Lung', str) is not None:
            return self.cancer_type_dic.TypetoID("Lung")
        elif re.search('Panc', str) is not None:
            return self.cancer_type_dic.TypetoID("Panc")
        elif re.search('NSCLC', str) is not None:
            return self.cancer_type_dic.TypetoID("NSCLC")
        else:
            return self.cancer_type_dic.TypetoID("Unknown")

    def plot(self):
        x_list = list(range(10))
        y_list = [0]*10
        for row_id, row in self.data.iterrows():
            index = row['Label']
            y_list[index] += 1
        # plot
        fig, ax = plt.subplots()
        ax.bar(x_list, y_list, width=1, edgecolor="white", linewidth=0.7, tick_label = [self.cancer_type_dic.IDtoType(x) for x in range(10)])
        plt.show()


if __name__ == '__main__':
    gse_data = GSEData()
    # df_train, df_test, df_val = gse_data.generate()
    gse_data.plot()