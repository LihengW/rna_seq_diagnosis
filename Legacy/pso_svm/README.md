## PSO-SVM

#### preparing the dataset

Put the dataset in the folder named "data". The specific format should be strictly similar with the "data.csv" and "label.csv"

#### adjust the parameters

The parameters of PSO can be adjusted by the variable "options"

`options = {'c1': 0.5 + math.log(2), 'c2': 0.5 + math.log(2), 'w': 1/(2 * math.log(2)), 'k': 3, 'p': 2}`

The parameters of SVM can be adjusted in "Classifier.py"

```python
model = svm.SVC(C=3.0,
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
```

#### Run the code

If you want to have a simple training with pso-svm, you can directly run function pso_svm(). 

If you want to have a n-fold-cross-verification, you can use function n_fold_pso_svm().

the outcome showed in data folder contains two part. The first one is the accuracy of all folds and the second one is the selected features by PSO. **If you use pso_svm(), then you have to get the return value with your own variable. So when wanting a clear result,  n_fold_pso_svm() is more recommended even n=1.**
