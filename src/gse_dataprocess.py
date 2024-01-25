
import pandas as pd
import re
import matplotlib.pyplot as plt
from utility.data_func import CancerTypeDic


def rdata_to_csv():
  import rpy2.robjects as robjects
  import rpy2.robjects as ro
  from rpy2.robjects import pandas2ri
  a = robjects.r['load']("RNAData/GSE183635/GSE183635_TEP_Count_Matrix.Rdata")
  k = robjects.r['TEP_Count_Matrix']
  with (ro.default_converter + pandas2ri.converter).context():
    df = ro.conversion.get_conversion().rpy2py(k)
  df = pd.DataFrame(df)
  df.index = k.rownames
  df.columns = k.colnames
  df = df.T
  df.to_csv("RNAData/GSE183635/GSE183635_Matrix.csv")


class GSE183635Data:
  def __init__(self):
    self.data = pd.read_csv('RNAData/GSE183635/GSE183635_Matrix.csv')
    self.data.index = self.data.loc[:, "sampleId"]
    self.data.drop(labels="sampleId", axis=1, inplace=True)

    self.labels = set()
    self.cancer_type_dic = CancerTypeDic()
    for index in self.data.index:
      cate = self.fill_in_cates(index, 'GSE183635')
      if cate != -1:
        self.data.loc[index, "Organ"] = cate
        self.labels.add(self.cancer_type_dic.IDtoType(cate, "GSE183635"))
      else:
        self.data.drop(labels=index, inplace=True)

    self.data.rename(columns={'sampleId': 'Id', 'Organ': 'Label'}, inplace=True)

  def fill_in_cates(self, str, dict_name):
    if re.search('HD', str) is not None or re.search('Control', str) is not None or re.search('HC', str) is not None :
      return self.cancer_type_dic.TypetoID("Health", dict_name)
    elif re.search('Breast', str) is not None or re.search('BrCa', str) is not None or re.search('BrCA', str) is not None:
      return self.cancer_type_dic.TypetoID("Breast", dict_name)
    elif re.search('Chol', str) is not None or re.search('CHOL', str) is not None:
      return self.cancer_type_dic.TypetoID("Chol", dict_name)
    elif re.search('CRC', str) is not None:
      return self.cancer_type_dic.TypetoID("CRC", dict_name)
    elif re.search('GBM', str) is not None:
      return self.cancer_type_dic.TypetoID("GBM", dict_name)
    elif re.search('Panc', str) is not None or re.search('PANC', str) is not None:
      return self.cancer_type_dic.TypetoID("Panc", dict_name)
    elif re.search('NSCLC', str) is not None:
      return self.cancer_type_dic.TypetoID("NSCLC", dict_name)
    elif re.search('HIV', str) is not None:
      return self.cancer_type_dic.TypetoID("HIV", dict_name)
    elif re.search('ProsBlad', str) is not None:
      return self.cancer_type_dic.TypetoID("ProsBlad", dict_name)
    elif re.search('PDAC', str) is not None:
      return self.cancer_type_dic.TypetoID("PDAC", dict_name)
    elif re.search('headNeck', str) is not None:
      return self.cancer_type_dic.TypetoID("headNeck", dict_name)
    elif re.search('HNSCC', str) is not None:
      return self.cancer_type_dic.TypetoID("HNSCC", dict_name)
    elif re.search('MELA', str) is not None:
      return self.cancer_type_dic.TypetoID("MELA", dict_name)
    elif re.search('OVA', str) is not None or re.search('OVARY', str) is not None or re.search('Ovary', str) is not None or re.search('Ova', str) is not None:
      return self.cancer_type_dic.TypetoID("OVA", dict_name)
    elif re.search('ORC', str) is not None:
      return self.cancer_type_dic.TypetoID("ORC", dict_name)
    elif re.search('MM', str) is not None:
      return self.cancer_type_dic.TypetoID("MM", dict_name)
    elif re.search('Sarc', str) is not None or re.search('SARC', str) is not None:
      return self.cancer_type_dic.TypetoID("Sarc", dict_name)
    elif re.search('BRMETA', str) is not None or re.search('Brain', str) is not None:
      return self.cancer_type_dic.TypetoID("BRMETA", dict_name)
    elif re.search('ChronPan', str) is not None or re.search('chPan', str) is not None:
      return self.cancer_type_dic.TypetoID("ChronPan", dict_name)
    elif re.search('IPMN', str) is not None:
      return self.cancer_type_dic.TypetoID("IPMN", dict_name)
    elif re.search('EPI', str) is not None:
      return self.cancer_type_dic.TypetoID("EPI", dict_name)
    elif re.search('UMCG', str) is not None:
      return self.cancer_type_dic.TypetoID("UMCG", dict_name)
    elif re.search('MGH', str) is not None:
      return self.cancer_type_dic.TypetoID("MGH", dict_name)
    elif re.search('HN-VU', str) is not None:
      return self.cancer_type_dic.TypetoID("HN-VU", dict_name)
    elif re.search('URO', str) is not None or re.search('Uro', str) is not None:
      return self.cancer_type_dic.TypetoID("URO", dict_name)
    elif re.search('HL', str) is not None:
      return self.cancer_type_dic.TypetoID("HL", dict_name)
    elif re.search('MS', str) is not None:
      return self.cancer_type_dic.TypetoID("MS", dict_name)
    elif re.search('LGG', str) is not None:
      return self.cancer_type_dic.TypetoID("LGG", dict_name)
    elif re.search('Pca', str) is not None:
      return self.cancer_type_dic.TypetoID("Pca", dict_name)
    elif re.search('PAAD', str) is not None:
      return self.cancer_type_dic.TypetoID("PAAD", dict_name)
    else:
      return self.cancer_type_dic.TypetoID("Unknown", dict_name)

  def plot(self):
    class_num = self.cancer_type_dic.GSE183635_class_num
    x_list = list(range(class_num))
    y_list = [0] * (class_num)
    for row_id, row in self.data.iterrows():
      index = row['Label']
      y_list[index] += 1
    max = 0
    for i in range(len(y_list)):
      if max < y_list[i]:
        max = y_list[i]
    explode = [max / (x * 200) for x in y_list]
    plt.figure(dpi=600)
    plt.pie(x=y_list,
            labels=[self.cancer_type_dic.IDtoType(x, "GSE183635") for x in range(class_num)],
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
      # plot
      # fig, ax = plt.subplots()
      # ax.bar(x_list, y_list, width=1, edgecolor="white", linewidth=0.7,
      #        tick_label=[self.cancer_type_dic.IDtoType(x, "GSE183635") for x in range(class_num - 1)])
      # plt.show()

  def filter(self, num):
    class_num = self.cancer_type_dic.GSE183635_class_num
    cate_num = [0] * class_num
    cate_index = []
    for i in range(class_num):
      cate_index.append([])

    for row_id, row in self.data.iterrows():
      index = row['Label']
      cate_num[index] += 1
      cate_index[index].append(row_id)

    remove_cate_list = []
    remove_row_ids = []
    for i in range(class_num):
      if cate_num[i] < num:
        remove_cate_list.append(i)
        remove_row_ids += cate_index[i]

    self.data = self.data.drop(remove_row_ids)
    print("Remove following cates")
    print(remove_cate_list)
    return self.data.shape


if __name__ == '__main__':
  gse_data = GSE183635Data()
  gse_data.filter(num=80)
