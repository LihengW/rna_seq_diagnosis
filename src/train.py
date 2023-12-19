import torch
import model
import dataset
import numpy as np
from Utility.plot import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp
def train(model, optimizer, data, criterion, scheduler):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      scheduler.step()  # lr_scheduler updates
      return loss

def test(model, data):
      model.eval()
      out = model(data)
      pred = out[data.test_mask].argmax(dim=1) # Use the class with highest probability.
      tar = data.y[data.test_mask].argmax(dim=1)
      class_num = 8

      # roc_curving
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      for i in range(class_num):
            fpr[i], tpr[i], _ = roc_curve(data.y[data.test_mask][:, i].detach().numpy(), out[data.test_mask][:, i].detach().numpy(), pos_label=float())
            roc_auc[i] = auc(fpr[i], tpr[i])

      fpr["micro"], tpr["micro"], _ = roc_curve(data.y[data.test_mask][:, i].detach().numpy().ravel(), out[data.test_mask][:, i].detach().numpy().ravel())
      roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

      all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
      mean_tpr = np.zeros_like(all_fpr)
      for i in range(class_num):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
      mean_tpr /= class_num
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
      colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
      for i, color in zip(range(class_num), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

      plt.plot([0, 1], [0, 1], 'k--', lw=lw)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Some extension of Receiver operating characteristic to multi-class')
      plt.legend(loc="lower right")
      plt.show()

      # confusion matrix
      test_correct = pred == tar  # Check against ground-truth labels.
      cm = np.zeros((class_num,class_num), dtype=int)
      for x in range(len(pred)):
            idx_1, idx_2 = int(tar[x]), int(pred[x])
            cm[idx_1][idx_2] += 1
      plot_confusion_matrix(cm, range(class_num))

      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


if __name__ == '__main__':

      data = dataset.GraphData_Hust().data
      model = model.GAT()
      # criterion = torch.nn.MSELoss(reduction='mean')  # Define loss criterion.
      criterion = torch.nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      res = []
      for epoch in range(1, 40000):
          loss = train(model, optimizer, data, criterion, scheduler)
          res.append(float(loss))
          print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
      import matplotlib.pyplot as plt
      plt.plot(range(len(res)), res)
      plt.show()
      print(test(model, data))

