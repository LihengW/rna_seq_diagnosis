from train import Program
import rna_model
import dataset
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


graph_data = dataset.GraphData_Hust()
learning_rate = [0.001, 0.005, 0.01]
dropout = 0.25
epochs = [150000, 200000, 300000]

programs = []
# mlp1 = rna_model.FLATMLP(in_features=graph_data.num_features, hidden_channels=4096, out_features=2048, layernum=2, dropout=dropout)
# gat = rna_model.DownSamplingGATBlock(
#         in_features=mlp.out_features, out_features=graph_data.num_classes,
#         top_hidden_channels=mlp.out_features, in_head=4,
#         dropout=dropout, attention_dropout=dropout, layernum=2
# )
# mlp2 = rna_model.FLATMLP(in_features=mlp1.out_features, hidden_channels=2048, out_features=2048, layernum=4, dropout=dropout)
# mlp3 = rna_model.DownSamplingMLP(in_features=2048, top_hidden_channels=2048, out_features=512, layernum=4, dropout=dropout)
# sage = rna_model.SAGEBlock(in_features=mlp3.out_features, out_features=graph_data.num_classes, top_hidden_channels=mlp2.out_features, dropout=dropout, layernum=4)
# resgated = rna_model.ResGatedBlock(in_features=mlp3.out_features, out_features=graph_data.num_classes, top_hidden_channels=int(mlp2.out_features / 2), dropout=0.3, layernum=4)
#
# mlp12 = rna_model.MixedModel(mlp1, mlp2)
# mlp = rna_model.MixedModel(mlp12, mlp3)
# sage_mixed = rna_model.MixedModel(mlp, sage).to(device)
# resgated_mixed = rna_model.MixedModel(mlp, resgated).to(device)
#
# sage_thin = rna_model.SAGEBlock(in_features=mlp1.out_features, out_features=graph_data.num_classes, top_hidden_channels=mlp1.out_features, dropout=dropout, layernum=2)
# sage_mixed_thin = rna_model.MixedModel(mlp1, sage_thin)
# create a set of mlp

Sagenet = rna_model.SAGENET(in_features=graph_data.num_features, out_features=graph_data.num_classes)

for j in range(1):
    name = "SageNet"
    new_prog = Program(epoch=30000,
                       model=Sagenet,
                       learning_rate=0.001,
                       criterion=torch.nn.CrossEntropyLoss(),
                       program_name=name,
                       optimizer="SGD")
    programs.append(new_prog)

    # name = "RESGATED"
    # new_prog = Program(epoch=180000,
    #                    model=resgated_mixed,
    #                    learning_rate=0.00005,
    #                    criterion=torch.nn.CrossEntropyLoss(),
    #                    program_name=name)
    # programs.append(new_prog)
    # name = "SAGE"
    # new_prog = Program(epoch=120000,
    #                    model=sage_mixed,
    #                    learning_rate=0.0005,
    #                    criterion=torch.nn.CrossEntropyLoss(),
    #                    program_name=name)
    # programs.append(new_prog)

# for lr in learning_rate:
for prog in programs:
    prog.run()
    prog.roc_curve(train_data=True,save_fig=True)
    prog.confusion_matrix(train_data=True, save_fig=True, normalized=True)
    prog.draw_loss()
    prog.calculate_acc()
    # print("svm_predict------------------")
    # prog.svm_predict()
    # print("raw_svm_predict------------------")
    # prog.svm_predict_raw()

