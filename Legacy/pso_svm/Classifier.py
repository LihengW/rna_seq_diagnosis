import pandas as pd 
import numpy as np
import src.HUST_dataprocess as HUST
from sklearn.metrics import precision_score
from src.utility.data_func import CancerTypeDic
import warnings
from Timer import Timer

warnings.filterwarnings('ignore')


class Classifier():
    def __init__(self, alpha=0.88):
        hust_data = HUST.HUSTData()
        train, test, val = hust_data.generate()
        # train, test, val = GSE.data_process()
        self.data = {'train': train, 'test': test, 'val': val}
        self.labels = hust_data.labels
        self.cate_num = len(hust_data.labels)
        self.dimensions = train.shape[1] - 1  # number of features
        self.alpha = alpha
        self.opt_mode = 'mul'  # if you want to use accuracy as an optimizing goal, should set this to 'ac'
        self.timer_svm = Timer()

        # current SVM and features
        self.svm_model = None
        self.features = None

        # if is_show:
        #     self.test_id = list(test.loc[:, 'Id'].copy())
        # if have_test:
        #     f_test, f_val = data_deal.test_data_process()
        #     self.final_test_id = list(f_test.loc[:, 'Id'].copy())
        #     self.test_data = (f_test, f_val)

    def f_per_particle(self, features, alpha, mode='ac'):
        total_features = self.dimensions
        self.timer_svm.start()
        ac = self.classify(features)
        self.timer_svm.end()
        fitness = (alpha * (1.0 - ac) + (1.0 - alpha) * (np.count_nonzero(features) / total_features))

        return fitness

    def f(self, x):
        alpha = self.alpha
        n_particles = x.shape[0]
        j = [self.f_per_particle(x[i], alpha, mode=self.opt_mode) for i in range(n_particles)]
        return j

    def classify(self, features='default', mode="train"):
        # use PSO binary mask to choose the features
        data = self.preProcess(phase=mode)
        if features == 'default':
            features = [1]*(data.shape[1]-1)
        features = [1] + list(features)
        features = list(map(bool, features))  # turn into bool map
        data = data.loc[:, features]
        # svm training
        res, model = self.svc_predict(data)
        # calculate the accuracy
        if mode == "train":
            ac = precision_score(list(self.data['val'].loc[:, "Label"]), res, average='micro')
            print('Subset performance: %.3f' % (ac))
            return ac

        elif mode == "test":
            hit = {}
            num = {}
            cm = np.array([[0]*self.cate_num]*self.cate_num)
            labels = list(self.labels)
            cancer_type_dic = CancerTypeDic()
            labelids = [cancer_type_dic.TypetoID(x) for x in labels]
            test = list(self.data['test'].loc[:, "Label"])
            for index in range(len(test)):
                # judge the right prediction or not
                lb = test[index]
                cm[labelids.index(lb)][labelids.index(res[index])] += 1
                if lb in num:
                    num[lb] += 1
                    if lb == res[index]:
                        hit[lb] += 1
                else:
                    num[lb] = 1
                    if lb == res[index]:
                        hit[lb] = 1
                    else:
                        hit[lb] = 0

            ac = num
            sum_hit, sum_num = 0, 0
            for key in ac.keys():
                sum_hit += hit[key]
                sum_num += num[key]
                ac[key] = hit[key]/num[key]

            total_ac = sum_hit/sum_num

            plot.plot_confusion_matrix(cm=cm, classes=labels, normalize=True)

            print('FinalTest performance: %.3f' % (total_ac))
            print('FinalTest performance detail: ' + str(ac))
            return total_ac, ac, cm

    def preProcess(self, phase='train'):
        if phase == 'train':
            df_train = self.data['train'].copy()
            df_val = self.data['val'].copy()

            df_val['Label'] = -1

            out = pd.concat([df_train, df_val])
            out = out.fillna(-1)
            return out

        elif phase == 'test':
            df_train = self.data['train'].copy()
            df_test = self.data['test'].copy()

            df_test['Label'] = -1

            out = pd.concat([df_train, df_test])
            out = out.fillna(-1)
            return out

    def svc_predict(self, data):
        import sklearn.svm as svm
        from sklearn.multiclass import OneVsRestClassifier
        single_model = svm.SVC(C=3.0,
                        kernel='linear',
                        degree=3,
                        gamma='auto',
                        coef0=0.0,
                        shrinking=True,
                        probability=True,
                        tol=0.001,
                        cache_size=200, class_weight=None,
                        verbose=False,
                        max_iter=-1,
                        decision_function_shape='ovr',
                        random_state=None)
        model = OneVsRestClassifier(single_model,)

        data.reset_index(drop=True)
        train = data[data['Label'] != -1]
        target = np.array(train.pop('Label'))

        test = data[data['Label'] == -1]
        test.drop(['Label'], axis=1, inplace=True)

        # print('开始训练...')
        model.fit(train, target)
        # print('开始预测...')
        y_pred = model.predict(test)
        # y_pred = model.predict(test)
        # print('返回并写入...')

        return y_pred, model

    def time_print(self):
        self.timer_svm.write_time("SVM")


if __name__ == '__main__':
    classification = Classifier()
    print(classification.classify())
    #print(classification.classify([]))


