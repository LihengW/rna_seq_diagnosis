import pandas as pd
import re
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from utility.data_func import CancerTypeDic

class GSEData:
    def __init__(self):
        self.data = pd.read_csv('RNAData/GSE/GSE68086_TEP_data_matrix.csv')
        # self.data = pd.read_csv('/RNAData/GSE/GSE_recon.csv')
        self.data = self.data.T
        self.data.columns = self.data.loc["sampleId"]
        self.data.drop(labels="sampleId", inplace=True)

        self.labels = set()
        self.cancer_type_dic = CancerTypeDic()
        for index in self.data.index:
            cate = self.fill_in_cates(index, 'GSE')
            if cate != -1:
                self.data.loc[index, "Organ"] = cate
                self.labels.add(self.cancer_type_dic.IDtoType(cate, "GSE"))
            else:
                self.data.drop(labels=index, inplace=True)


        self.data.rename(columns={'sampleId': 'Id', 'Organ': 'Label'}, inplace=True)

    def generate(self):
        df_train, df_test = train_test_split(self.data, test_size=0.4, stratify=self.data['Label'])
        df_test, df_val = train_test_split(self.data, test_size=0.5, stratify=self.data['Label'])

        return df_train, df_test, df_val

    def fill_in_cates(self, str, dict_name):
        if re.search('HD', str) is not None or re.search('Control', str) is not None:
            return self.cancer_type_dic.TypetoID("Health", dict_name)
        elif re.search('Breast', str) is not None or re.search('BrCa', str) is not None:
            return self.cancer_type_dic.TypetoID("Breast", dict_name)
        elif re.search('Liver', str) is not None:
            return self.cancer_type_dic.TypetoID("Liver", dict_name)
        elif re.search('Chol', str) is not None:
            return self.cancer_type_dic.TypetoID("Chol", dict_name)
        elif re.search('CRC', str) is not None:
            return self.cancer_type_dic.TypetoID("CRC", dict_name)
        elif re.search('GBM', str) is not None:
            return self.cancer_type_dic.TypetoID("GBM", dict_name)
        elif re.search('Platelet', str) is not None:
            return self.cancer_type_dic.TypetoID("Platelet", dict_name)
        elif re.search('Lung', str) is not None:
            return self.cancer_type_dic.TypetoID("Lung", dict_name)
        elif re.search('Panc', str) is not None:
            return self.cancer_type_dic.TypetoID("Panc", dict_name)
        elif re.search('NSCLC', str) is not None:
            return self.cancer_type_dic.TypetoID("NSCLC", dict_name)
        else:
            return self.cancer_type_dic.TypetoID("Unknown", dict_name)

    def plot(self):
        # x_list = list(range(10))
        # y_list = [0]*10
        # for row_id, row in self.data.iterrows():
        #     index = row['Label']
        #     y_list[index] += 1
        # fig, ax = plt.subplots()
        # ax.bar(x_list, y_list, width=1, edgecolor="white", linewidth=0.7, tick_label = [self.cancer_type_dic.IDtoType(x, "GSE") for x in range(10)])
        # plt.show()

        class_num = self.cancer_type_dic.GSE_class_num
        x_list = list(range(class_num))
        y_list = [0] * (class_num)
        for row_id, row in self.data.iterrows():
            index = row['Label']
            y_list[index] += 1
        max = 0
        for i in range(len(y_list)):
            if max < y_list[i]:
                max = y_list[i]
        explode = [ max / (x*100) for x in y_list]

        plt.figure(dpi=600)
        plt.pie(x=y_list,
                labels=[self.cancer_type_dic.IDtoType(x, "GSE") for x in range(class_num)],
                explode=explode,
                radius=1,
                autopct='%.2f%%',
                textprops={'fontsize': 3},
                pctdistance=1.1,
                labeldistance=1.2,
                wedgeprops={'linestyle': '--', 'linewidth': 1}
                )
        plt.legend(loc='best', fontsize=2)
        plt.show()


if __name__ == '__main__':
    gse_data = GSEData()
    gse_data.plot()