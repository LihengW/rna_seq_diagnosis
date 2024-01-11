import pyreadr

dict = pyreadr.read_r('GSE183635_TEP_Count_Matrix.RData')
data = dict['TEP_Count_Matrix']
print(data)