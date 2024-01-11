import sklearn.svm as svm
from sklearn.multiclass import OneVsRestClassifier

class SVM:
    def __init__(self):
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

        self.single_model = single_model
        self.model = OneVsRestClassifier(self.single_model, n_jobs=-1)

    def fit(self, train_data, target):
        self.model.fit(train_data, target)

    def pred(self, test_data):
        return self.model.predict(test_data)