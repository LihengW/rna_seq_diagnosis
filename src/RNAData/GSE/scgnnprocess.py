import pandas as pd

data = pd.read_csv('/RNAData/GSE/GSE68086_TEP_data_matrix_gpu.csv')
data = data.T
data.to_csv('F:/RNA_seq/RNAData/GSE/GSE68086_TEP_data_matrix_gpu_t.csv')