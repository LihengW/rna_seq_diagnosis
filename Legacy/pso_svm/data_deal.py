import pandas as pd
from sklearn.model_selection import train_test_split


def data_process():  # changed
    data = pd.read_csv('data/data.csv')
    cate = pd.read_csv('data/label.csv')

    cate.loc[cate['Organ'] == 'HD', 'Organ'] = 0
    cate.loc[cate['Organ'] == 'stomach', 'Organ'] = 1
    cate.loc[cate['Organ'] == 'colon', 'Organ'] = 2
    cate.loc[cate['Organ'] == 'kidney', 'Organ'] = 3
    cate.loc[cate['Organ'] == 'pancreas', 'Organ'] = 4
    cate.loc[cate['Organ'] == 'lung', 'Organ'] = 5
    cate.loc[cate['Organ'] == 'breast', 'Organ'] = 6


    ndata = pd.merge(cate, data, how="left", on="sampleId")
    ndata.rename(columns={'sampleId': 'Id', 'Organ': 'Label'}, inplace=True)

    df_train, df_test = train_test_split(ndata, test_size=0.2, stratify=ndata['Label'])

    # df_tval is the labels belonging to test part
    df_tval = df_test.copy().loc[:, 'Id':'Label']
    df_test.drop(columns='Label', inplace=True)

    return df_train, df_test, df_tval


def test_data_process():  # not changed yet
    test_data = pd.read_csv('data/test_data.csv')
    test_cate = pd.read_csv('data/test_label.csv')

    test_cate.loc[test_cate['Type'] == 'cancer', 'Type'] = 1
    test_cate.loc[test_cate['Type'] == 'normal', 'Type'] = 0

    ndata = pd.merge(test_cate, test_data, how="left", on="sampleId")
    ndata.rename(columns={'sampleId': 'Id', 'Type': 'Label'}, inplace=True)

    test_val = ndata.copy().loc[:, 'Id':'Label']
    ndata.drop(columns='Label', inplace=True)


    return ndata, test_val


if __name__ == '__main__':
    train, test, val = data_process()
    print(train)



