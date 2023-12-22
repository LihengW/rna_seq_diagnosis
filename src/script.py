from train import Program
import rna_model
import dataset

graph_data = dataset.GraphData_Hust()
learning_rate = [0.001, 0.005, 0.01]
# model = rna_model.DownSamplingMLP(graph_data.num_features, 256, 4, graph_data.num_classes, 3)
mlp = rna_model.DownSamplingMLP(in_features=graph_data.num_features, top_hidden_channels=1024,
                                shrink_rate=2, out_features=256, layernum=3, dropout=0.4)

gat = rna_model.DownSamplingGATBlock(mlp.out_features,graph_data.num_classes, top_hidden_channels=256, in_head=4)

model = rna_model.MixedModel(mlp, gat)

programs = []
for lr in learning_rate:
    new_prog = Program(data=graph_data.data,
                       epoch=50000,
                       model=model,
                       learning_rate=lr,
                       program_name="DownSample")
    new_prog.train()
    new_prog.roc_curve(train_data=True,save_fig=True)
    new_prog.confusion_matrix(train_data=True, save_fig=True, normalized=True)

