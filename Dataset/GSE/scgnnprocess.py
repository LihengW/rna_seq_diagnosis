import pandas as pd

data = pd.read_csv('F:/RNA_seq/Dataset/GSE/GSE68086_TEP_data_matrix_gpu.csv')
data = data.T
data.to_csv('F:/RNA_seq/Dataset/GSE/GSE68086_TEP_data_matrix_gpu_t.csv')