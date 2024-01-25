import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def scplot():
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()

    sc.settings.set_figure_params(dpi=200,
                                  dpi_save=400,
                                  fontsize=8,
                                  facecolor='white')
    # # Local DATA
    # adata = sc.read_csv('RNAData/HUST/NewHust_ReconGenes.csv')
    # label = pd.read_csv('RNAData/HUST/HustLabels.csv')
    #
    # # GSE68086
    # adata2 = sc.read_csv('RNAData/GSE/GSE68086_data_matrix.csv')
    # label2 = pd.read_csv('RNAData/GSE/GSE68086_Labels.csv')

    # GSE18363635
    adata = sc.read_csv('RNAData/GSE183635/GSE183635Data_without_label.csv')
    label = pd.read_csv('RNAData/GSE183635/GSE183635Label.csv')


    label.index = label.iloc[:, 0]
    label.drop(labels='ID', axis=1, inplace=True)
    sc.pl.highest_expr_genes(adata, n_top=30, log=False)
    adata.obs = label

    sc.pp.filter_genes(adata, min_cells=10)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], log1p=False, inplace=True)

    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'],
                 jitter=0.4, multi_panel=True)

    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata)

    adata.raw = adata

    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts'])
    sc.pp.scale(adata, max_value=10)


    sc.tl.pca(adata, svd_solver='arpack', n_comps=40)
    sc.pl.pca(adata, color='Label')

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.louvain(adata, resolution=1.0, )
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos='paga')
    sc.pl.umap(adata, color=['Label', 'louvain'])
    sc.pl.umap(adata, color=['Label', 'louvain'], use_raw=False)

    sc.tl.rank_genes_groups(adata, 'louvain', method='t-test')
    # sc.pl.rank_genes_groups(adata, n_genes=50, sharey=False)

    sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)


    sc.pl.violin(adata, ['Label'], groupby='louvain')


def basic_plot():
    from HUST_dataprocess import NewHustData
    hust_data = NewHustData()
    from TEP_dataprocess import GSEData
    gse_data = GSEData()
    from gse_dataprocess import GSE183635Data
    gse_data2 = GSE183635Data()

    hust_data_gene = set(hust_data.data.columns[1:])
    gse_data_gene = set(gse_data.data.columns[1:])
    gse_data2_gene = set(gse_data2.data.columns[1:])

    gene_sets = [hust_data_gene, gse_data_gene, gse_data2_gene]

    print("HUST:  " + str(len(hust_data_gene)))
    print("GSE:  " + str(len(gse_data_gene)))
    print("GSE2:  " + str(len(gse_data2_gene)))

    overlap_matrix = np.zeros((3, 3))

    for background_index in range(3):
        for chosen_index in range(3):
            if background_index == chosen_index:
                overlap_matrix[chosen_index][chosen_index] = len(gene_sets[chosen_index])
            else:
                for gene in gene_sets[chosen_index]:
                    if gene in gene_sets[background_index]:
                        overlap_matrix[background_index][chosen_index] += 1

    print(overlap_matrix)


def pearson_plot():
    from HUST_dataprocess import NewHustData
    hust_data = NewHustData()
    from TEP_dataprocess import GSEData
    gse_data = GSEData()
    from gse_dataprocess import GSE183635Data
    gse_data2 = GSE183635Data()

    df = hust_data.data.drop(columns="Label")
    pearson = np.array(df.corr(method="pearson"))
    pearson = pearson.ravel().T

    fig = plt.figure()

    ax = plt.subplot(311)
    n, bins_num, pat = ax.hist(pearson, bins=50, alpha=0.75, rwidth=0.7, color='green')
    ax.plot(bins_num[:-1], n, marker='o', color="yellowgreen", linestyle="--")
    ax.set_title("Local")

    df = gse_data.data.drop(columns="Label")
    pearson = np.array(df.corr(method="pearson"))
    pearson = pearson.ravel().T

    ax = plt.subplot(312)
    n, bins_num, pat = ax.hist(pearson, bins=50, alpha=0.75, rwidth=0.7, color='green')
    ax.plot(bins_num[:-1], n, marker='o', color="yellowgreen", linestyle="--")
    ax.set_title("GSE68086")

    df = gse_data2.data.drop(columns="Label")
    pearson = np.array(df.corr(method="pearson"))
    pearson = pearson.ravel().T

    ax = plt.subplot(313)
    n, bins_num, pat = ax.hist(pearson, bins=50, alpha=0.75, rwidth=0.7, color='green')
    ax.plot(bins_num[:-1], n, marker='o', color="yellowgreen", linestyle="--")
    ax.set_title("GSE183635")

    plt.show()


def public_gene_pearson_plot():
    from HUST_dataprocess import NewHustData
    hust_data = NewHustData()
    from TEP_dataprocess import GSEData
    gse_data = GSEData()
    from gse_dataprocess import GSE183635Data
    gse_data2 = GSE183635Data()

    from dataset import PublicGeneList
    public_gene_list = PublicGeneList().get()

    df = hust_data.data.drop(columns="Label")
    df = df.loc[:, public_gene_list]
    pearson = np.array(df.corr(method="pearson"))
    pearson = pearson.ravel().T

    fig = plt.figure()

    ax = plt.subplot(311)
    n, bins_num, pat = ax.hist(pearson, bins=50, alpha=0.75, rwidth=0.7, color='red')
    ax.plot(bins_num[:-1], n, marker='o', color="orange", linestyle="--")
    ax.set_title("Local")

    df = gse_data.data.drop(columns="Label")
    df = df.loc[:, public_gene_list]
    pearson = np.array(df.corr(method="pearson"))
    pearson = pearson.ravel().T

    ax = plt.subplot(312)
    n, bins_num, pat = ax.hist(pearson, bins=50, alpha=0.75, rwidth=0.7, color='red')
    ax.plot(bins_num[:-1], n, marker='o', color="orange", linestyle="--")
    ax.set_title("GSE68086")

    df = gse_data2.data.drop(columns="Label")
    df = df.loc[:, public_gene_list]
    pearson = np.array(df.corr(method="pearson"))
    pearson = pearson.ravel().T

    ax = plt.subplot(313)
    n, bins_num, pat = ax.hist(pearson, bins=50, alpha=0.75, rwidth=0.7, color='red')
    ax.plot(bins_num[:-1], n, marker='o', color="orange", linestyle="--")
    ax.set_title("GSE183635")

    plt.show()





if __name__ == '__main__':
    public_gene_pearson_plot()