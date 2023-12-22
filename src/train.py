import torch
import rna_model
import dataset
import numpy as np
from Utility.plot import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt


class Program:
    def __init__(self,
                 data=dataset.GraphData_Hust().data,
                 epoch=50000,
                 model=rna_model.GAT(),
                 criterion=torch.nn.CrossEntropyLoss(),
                 optimizer="SGD",
                 learning_rate=0.01,
                 class_num=8,
                 program_name="program",
                 ):
        self.data = data
        self.epoch = epoch
        self.model = model
        self.class_num = class_num
        # criterion = torch.nn.MSELoss(reduction='mean')  # Define loss criterion.
        self.criterion = criterion
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.program_name = program_name

    def run(self):
        res = []
        for epoch in range(1, self.epoch):
            loss = self.train()
            res.append(float(loss))
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        import matplotlib.pyplot as plt
        plt.plot(range(len(res)), res)
        plt.show()
        print(self.test())

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(self.data)  # Perform a single forward pass.
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])  # Compute the loss solely based on the training nodes.
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
        test_pred = out[self.data.test_mask].detach().numpy()
        test_tar = self.data.y[self.data.test_mask].detach().numpy()

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
            train_pred = out[self.data.train_mask].detach().numpy()
            train_tar = self.data.y[self.data.train_mask].detach().numpy()

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
        test_pred = out[self.data.test_mask].argmax(dim=1)
        test_tar = self.data.y[self.data.test_mask].argmax(dim=1)
        test_correct = test_pred == test_tar  # Check against ground-truth labels.

        cm = np.zeros((self.class_num, self.class_num), dtype=int)
        for x in range(len(test_pred)):
            idx_1, idx_2 = int(test_tar[x]), int(test_pred[x])
            cm[idx_1][idx_2] += 1
        plot_confusion_matrix(cm, range(self.class_num), title="TEST Result", cmap=plt.get_cmap('cool'), show=not save_fig)
        if save_fig:
            plt.savefig("figs/" + self.program_name + "_test_cm" + ".png", bbox_inches='tight', dpi=600, transparent=True)
            plt.clf()
        else:
            plt.show()
        if normalized:
            plot_confusion_matrix(cm, range(self.class_num), title="TEST Result", cmap=plt.get_cmap('coolwarm'), show=not save_fig, normalize=True)
            if save_fig:
                plt.savefig("figs/" + self.program_name + "_test_cm_normalized" + ".png", bbox_inches='tight', dpi=600,
                            transparent=True)
                plt.clf()
            else:
                plt.show()
        test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())  # Derive ratio of correct predictions.

        if train_data:
            train_pred = out[self.data.train_mask].argmax(dim=1)
            train_tar = self.data.y[self.data.train_mask].argmax(dim=1)
            train_correct = train_pred == train_tar  # Check against ground-truth labels.

            cm = np.zeros((self.class_num, self.class_num), dtype=int)
            for x in range(len(train_pred)):
                idx_1, idx_2 = int(train_tar[x]), int(train_pred[x])
                cm[idx_1][idx_2] += 1
            plot_confusion_matrix(cm, range(self.class_num), title="TRAIN Result", cmap=plt.get_cmap('cool'),
                                  show=not save_fig)
            if save_fig:
                plt.savefig("figs/" + self.program_name + "_train_cm" + ".png", bbox_inches='tight', dpi=600,
                            transparent=True)
                plt.clf()
            else:
                plt.show()
            if normalized:
                plot_confusion_matrix(cm, range(self.class_num), title="TEST Result", cmap=plt.get_cmap('coolwarm'),
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
