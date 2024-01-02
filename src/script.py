from train import Program
import rna_model
import dataset
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

graph_data = dataset.GraphData_Hust()
# crossweight = torch.tensor([1, 2, 1, 1, 2, 1.5, 2, 2])
# crossweight.to(device)
learning_rate = [0.001, 0.005, 0.01]
dropout = 0.2
epochs = [150000, 200000, 300000]

# model = rna_model.DownSamplingMLP(graph_data.num_features, 256, 4, graph_data.num_classes, 3)
# mlp_dict = {
#             "3Layer1024Channel": rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=256, top_hidden_channels=1024,layernum=3, dropout=dropout),
#             "3Layer2048Channel": rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=512, top_hidden_channels=2048,layernum=3, dropout=dropout),
#             # "3Layer512Channel": rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=128, top_hidden_channels=512,layernum=3, dropout=dropout),
#             # "3Layer256Channel": rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=64, layernum=3, top_hidden_channels=256,dropout=dropout),
#             # "4Layer2048Channel": rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=256, layernum=4, top_hidden_channels=2048,dropout=dropout),
#             # "4Layer1024Channel": rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=128, layernum=4, top_hidden_channels=1024,dropout=dropout),
#             # "4Layer512Channel": rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=64, layernum=4, top_hidden_channels=512,dropout=dropout),
#             }
#
# mlp = rna_model.DownSamplingMLP(in_features=graph_data.num_features, top_hidden_channels=1024,
#                                 shrink_rate=2, out_features=256, layernum=3, dropout=0.4)

# gat = rna_model.DownSamplingGATBlock(mlp.out_features,graph_data.num_classes, top_hidden_channels=256, in_head=4)

# model = rna_model.MixedModel(mlp, gat)

programs = []
mlp = rna_model.DownSamplingMLP(in_features=graph_data.num_features, out_features=64, top_hidden_channels=2048,layernum=5, dropout=dropout)
# gat = rna_model.DownSamplingGATBlock(
#         in_features=mlp.out_features, out_features=graph_data.num_classes,
#         top_hidden_channels=mlp.out_features, in_head=4,
#         dropout=dropout, attention_dropout=dropout, layernum=2
# )
sage = rna_model.SAGEBlock(in_features=mlp.out_features, out_features=graph_data.num_classes, top_hidden_channels=int(mlp.out_features / 2), dropout=dropout, layernum=2)
resgated = rna_model.ResGatedBlock(in_features=mlp.out_features, out_features=graph_data.num_classes, top_hidden_channels=int(mlp.out_features / 2), dropout=dropout, layernum=2)

sage_mixed = rna_model.MixedModel(mlp, sage).to(device)
resgated_mixed = rna_model.MixedModel(mlp, resgated).to(device)

# create a set of mlp
for j in range(1):
    name = "5MLP2SAGE"
    new_prog = Program(epoch=80000,
                       model=sage_mixed,
                       learning_rate=learning_rate[2],
                       criterion=torch.nn.CrossEntropyLoss(),
                       program_name=name)
    programs.append(new_prog)
    name = "5MLP2RESGATED"
    new_prog = Program(epoch=80000,
                       model=resgated_mixed,
                       learning_rate=learning_rate[2],
                       criterion=torch.nn.CrossEntropyLoss(),
                       program_name=name)
    programs.append(new_prog)

# for lr in learning_rate:
for prog in programs:
    prog.run()
    prog.roc_curve(train_data=True,save_fig=True)
    prog.confusion_matrix(train_data=True, save_fig=True, normalized=True)
    prog.draw_loss()

