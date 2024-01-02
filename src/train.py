import torch
import rna_model
import dataset
import numpy as np
from utility import plot
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt


class Program:
    def __init__(self,
                 epoch=50000,
                 model=rna_model.GAT(),
                 criterion=torch.nn.CrossEntropyLoss(),
                 optimizer="SGD",
                 learning_rate=0.01,
                 class_num=8,
                 program_name="program",
                 loss_weighted=True
                 ):
        self.graph_data = dataset.GraphData_Hust()
        self.data = self.graph_data.data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data.to(self.device)
        self.epoch = epoch
        self.model = model
        self.model.to(self.device)
        self.class_num = class_num
        # criterion = torch.nn.MSELoss(reduction='mean')  # Define loss criterion.
        if loss_weighted:
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.graph_data.class_weight).to(self.device))
        else:
            self.criterion = criterion
        self.criterion.to(self.device)

        self.loss_record = []
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        self.program_name = program_name

    def run(self):
        for epoch in range(1, self.epoch):
            loss = self.train()
            print(f'{self.program_name}: --- Epoch: {epoch:03d}, Loss: {loss:.4f}, Process: {epoch/self.epoch:.4f}')

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(self.data)  # Perform a single forward pass.
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])  # Compute the loss solely based on the training nodes.
        self.loss_record.append(float(loss))
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        self.scheduler.step()  # lr_scheduler updates
        return loss


    def test(self):
        self.model.eval()
        out = self.model(self.data)
        pred = out[self.data.test_mask].argmax(dim=1)  # Use the class with highest probability.
        tar = self.data.y[self.data.test_mask].argmax(dim=1)
        return out[self.data.test_mask]

    def roc_curve(self, train_data=False, save_fig=False):
        # roc_curving
        self.model.eval()
        out = self.model(self.data)

        # Use the class with highest probability.
        # test_pred = out[self.data.test_mask].argmax(dim=1)
        # test_tar = self.data.y[self.data.test_mask].argmax(dim=1)

        # TEST GRAPH
        # Use the class score
        test_pred = out[self.data.test_mask].cpu().detach().numpy()
        test_tar = self.data.y[self.data.test_mask].cpu().detach().numpy()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.class_num):
            # REVERSE THE SCORE TO PERFORM BETTER
            fpr[i], tpr[i], _ = roc_curve(test_tar[:, i],
                                          1 - test_pred[:, i],
                                          pos_label=float())
            roc_auc[i] = auc(fpr[i], tpr[i])

        # MICRO ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(test_tar[:, i].ravel(),
                                                  (1 - test_pred[:, i].ravel()))
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # MACRO ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.class_num)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.class_num):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.class_num
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        lw = 2
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        from itertools import cycle
        colors = cycle(['forestgreen', 'aqua', 'darkorange',
                        'cornflowerblue', 'lightcoral', 'slateblue',
                        'darkmagenta', 'crimson', 'dodgerblue'])
        for i, color in zip(range(self.class_num), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.program_name + " TEST ROC")
        plt.legend(loc="lower right")

        if save_fig:
            plt.savefig("figs/" + self.program_name + "_test_roc" + ".png", bbox_inches='tight', dpi=600, transparent=True)
            plt.clf()
        else:
            plt.show() # draw the fig

        if train_data:
            # TRAIN GRAPH
            # Use the class score
            train_pred = out[self.data.train_mask].cpu().detach().numpy()
            train_tar = self.data.y[self.data.train_mask].cpu().detach().numpy()

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(self.class_num):
                # REVERSE THE SCORE TO PERFORM BETTER
                fpr[i], tpr[i], _ = roc_curve(train_tar[:, i],
                                              1 - train_pred[:, i],
                                              pos_label=float())
                roc_auc[i] = auc(fpr[i], tpr[i])

            # MICRO ROC
            fpr["micro"], tpr["micro"], _ = roc_curve(train_tar[:, i].ravel(),
                                                      (1 - train_pred[:, i].ravel()))
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # MACRO ROC
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.class_num)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.class_num):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= self.class_num
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            lw = 2
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            from itertools import cycle
            colors = cycle(['forestgreen', 'aqua', 'darkorange',
                            'cornflowerblue', 'lightcoral', 'slateblue',
                            'darkmagenta', 'crimson', 'dodgerblue'])
            for i, color in zip(range(self.class_num), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(self.program_name + " TRAIN ROC")
            plt.legend(loc="lower right")

            if save_fig:
                plt.savefig("figs/" + self.program_name + "_train_roc" + ".png",bbox_inches='tight', dpi=600, transparent=True)
                plt.clf()
            else:
                plt.show()  # draw the fig
    def confusion_matrix(self, train_data=False, save_fig=False, normalized=False):
        self.model.eval()
        out = self.model(self.data)
        # Use the class with the highest probability.
        test_pred = out[self.data.test_mask].cpu().argmax(dim=1)
        test_tar = self.data.y[self.data.test_mask].cpu().argmax(dim=1)
        test_correct = test_pred == test_tar  # Check against ground-truth labels.

        cm = np.zeros((self.class_num, self.class_num), dtype=int)
        for x in range(len(test_pred)):
            idx_1, idx_2 = int(test_tar[x]), int(test_pred[x])
            cm[idx_1][idx_2] += 1
        plot.plot_confusion_matrix(cm, range(self.class_num), title="TEST Result", cmap=plt.get_cmap('cool'), show=not save_fig)
        if save_fig:
            plt.savefig("figs/" + self.program_name + "_test_cm" + ".png", bbox_inches='tight', dpi=600, transparent=True)
            plt.clf()
        else:
            plt.show()
        if normalized:
            plot.plot_confusion_matrix(cm, range(self.class_num), title="TEST Result", cmap=plt.get_cmap('coolwarm'), show=not save_fig, normalize=True)
            if save_fig:
                plt.savefig("figs/" + self.program_name + "_test_cm_normalized" + ".png", bbox_inches='tight', dpi=600,
                            transparent=True)
                plt.clf()
            else:
                plt.show()
        test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())  # Derive ratio of correct predictions.

        if train_data:
            train_pred = out[self.data.train_mask].cpu().argmax(dim=1)
            train_tar = self.data.y[self.data.train_mask].cpu().argmax(dim=1)
            train_correct = train_pred == train_tar  # Check against ground-truth labels.

            cm = np.zeros((self.class_num, self.class_num), dtype=int)
            for x in range(len(train_pred)):
                idx_1, idx_2 = int(train_tar[x]), int(train_pred[x])
                cm[idx_1][idx_2] += 1
            plot.plot_confusion_matrix(cm, range(self.class_num), title="TRAIN Result", cmap=plt.get_cmap('cool'),
                                       show=not save_fig)
            if save_fig:
                plt.savefig("figs/" + self.program_name + "_train_cm" + ".png", bbox_inches='tight', dpi=600,
                            transparent=True)
                plt.clf()
            else:
                plt.show()
            if normalized:
                plot.plot_confusion_matrix(cm, range(self.class_num), title="TEST Result", cmap=plt.get_cmap('coolwarm'),
                                           show=not save_fig, normalize=True)
                if save_fig:
                    plt.savefig("figs/" + self.program_name + "_train_cm_normalized" + ".png", bbox_inches='tight',
                                dpi=600,
                                transparent=True)
                    plt.clf()
                else:
                    plt.show()
            train_acc = int(train_correct.sum()) / int(self.data.train_mask.sum())  # Derive ratio of correct predictions.

        return test_acc, train_acc

    def draw_loss(self):
        plt.plot(list(range(len(self.loss_record))), self.loss_record)
        plt.title("Loss")
        plt.tight_layout()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("figs/" + self.program_name + "loss_value", bbox_inches='tight', dpi=600, transparent=False)


if __name__ == '__main__':
    progame = Program()
    # train example
    # data = dataset.GraphData_Hust().data
    # model = model.GAT()
    # # criterion = torch.nn.MSELoss(reduction='mean')  # Define loss criterion.
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # res = []
    # for epoch in range(1, 4000):
    #     loss = train(model, optimizer, data, criterion, scheduler)
    #     res.append(float(loss))
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    # import matplotlib.pyplot as plt
    #
    # plt.plot(range(len(res)), res)
    # plt.show()
    # print(test(model, data))
