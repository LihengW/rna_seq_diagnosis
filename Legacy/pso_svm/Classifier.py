import pandas as pd 
import numpy as np
import Dataset.GSE.TEP_dataprocess as GSE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import data_deal
import warnings
from Timer import Timer

warnings.filterwarnings('ignore')


class Classifier():
    def __init__(self, type='svm', is_show=False, alpha=0.88, have_test=False,):
        train, test, val = data_deal.data_process()
        # train, test, val = GSE.data_process()
        self.data = {'train': train, 'test': test, 'val': val}
        self.dimensions = train.shape[1] - 2  # number of features
        self.type = type  # type of the classification
        self.alpha = alpha
        self.is_show = is_show  # print the prediction result
        self.opt_mode = 'mul'  # if you want to use accuracy as an optimizing goal, should set this to 'ac'
        self.timer_svm = Timer()
        if is_show:
            self.test_id = list(test.loc[:, 'Id'].copy())
        if have_test:
            f_test, f_val = data_deal.test_data_process()
            self.final_test_id = list(f_test.loc[:, 'Id'].copy())
            self.test_data = (f_test, f_val)

    def f_per_particle(self, features, alpha, mode='ac'):
        total_features = self.dimensions
        if mode == 'ac':
            self.timer_svm.start()
            ac = self.classify(features)[0]
            self.timer_svm.end()
            fitness = (alpha * (1.0 - ac) + (1.0 - alpha) * (np.count_nonzero(features) / total_features))
        else:
            mul_ac = self.classify(features)[1]
            parameters = [1, 3, 1, 2, 1, 3, 1]
            fitness = 0
            for cate in range(len(parameters)):
                fitness += (alpha * (1.0 - mul_ac[cate])) * (parameters[cate] / sum(parameters))
            fitness += (1.0 - alpha) * (np.count_nonzero(features) / total_features)

        return fitness

    def f(self, x):
        alpha = self.alpha
        n_particles = x.shape[0]
        j = [self.f_per_particle(x[i], alpha, mode=self.opt_mode) for i in range(n_particles)]
        return j

    def classify(self, features='default', is_show=False, have_test=False):
        # use PSO binary mask to choose the features
        data = self.preProcess(phase='train')
        val = self.data['val'].copy()
        if have_test:
            test_data = self.preProcess(phase='test')
            test_val = self.test_data[1].copy()

        if features == 'default':
            features = [1]*(data.shape[1]-1)

        if have_test:
            t_features = list(features) + [1]
            t_features = list(map(bool, t_features))  # turn into bool map
            test_data = test_data.loc[:, t_features]
        features = [1] + list(features)
        features = list(map(bool, features))  # turn into bool map
        data = data.loc[:, features]

        # get the result of classification
        if self.type == 'lr':
            # lr training
            continuous_feature = list(data.columns)[1:]
            res = self.lr_predict(data, continuous_feature)

        elif self.type == 'svm':
            # svm training
            res, model = self.svc_predict(data)
            if have_test:
                test_data.drop(['Label'], axis=1, inplace=True)
                test_res = model.predict_proba(test_data)[:, 1]

        else:
            raise NotImplementedError

        # calculate the accuracy
        if not have_test:
            err = [0]*7
            num = [0]*7
            mul_ac = [0]*7
            ac = 0
            val = list(val.iloc[:, 1])
            if len(val) != len(res):
                raise ValueError
            for index in range(len(val)):
                # judge the right prediction or not
                num[val[index]] += 1
                if val[index] != res[index]:
                    err[val[index]] += 1
            for label in range(7):
                mul_ac[label] = 1 - (err[label] / num[label])

            ac = 1 - (sum(err)/sum(num))

            print('Subset performance: %.3f' % (ac))
            print('Subset performance detail: ' + str(mul_ac))

            return (ac, mul_ac, res)

        else:
            err = 0
            val = test_val
            res = list(test_res)
            val = list(val.iloc[:, 1])
            if len(val) != len(res):
                raise ValueError
            for index in range(len(val)):
                if val[index] != res[index]:
                    err += 1
            ac = 1 - (err / len(val))

            print('Test performance: %.3f' % (ac))
            return (ac, res)

    def preProcess(self, phase='train'):
        if phase == 'train':
            df_train = self.data['train'].copy()
            df_test = self.data['test'].copy()

            df_train.drop(['Id'], axis=1, inplace=True)
            df_test.drop(['Id'], axis=1, inplace=True)

            df_test['Label'] = -1

            out = pd.concat([df_train, df_test])
            out = out.fillna(-1)
            return out

        elif phase == 'test':
            df_test = self.test_data[0].copy()
            df_test.drop(['Id'], axis=1, inplace=True)
            df_test['Label'] = -1

            return df_test

    def lr_predict(self, data, continuous_feature): # 0.47181
        # 连续特征归一化
        print('开始归一化...')
        scaler = MinMaxScaler()
        for col in continuous_feature:
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        print('归一化结束')

        train = data[data['Label'] != -1]
        target = train.pop('Label')
        test = data[data['Label'] == -1]
        test.drop(['Label'], axis=1, inplace=True)

        # 划分数据集
        x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)
        # lr classification
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_pred = lr.predict_proba(test)[:, 1]

        return y_pred

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


