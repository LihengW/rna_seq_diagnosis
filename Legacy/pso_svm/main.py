import math
from Classifier import Classifier
import pyswarms as ps
import pandas as pd
import numpy as np
from Timer import Timer

Label_match = {0: 'Health', 1: 'cancer in stomach', 2: 'cancer in colon', 3: 'cancer in kidney',
               4: 'cancer in pancreas', 5: 'cancer in lung', 6: 'cancer in breast'}


def pso_svm(options='default', n_fold=False, is_show=False, have_test=False):
    # Initialize the swarm, arbitrary
    if options == 'default':
        options = {'c1': 0.5 + math.log(2), 'c2': 0.5 + math.log(2), 'w': 1/(2 * math.log(2)), 'k': 3, 'p': 2}

    # create the classification instance
    classifier = Classifier(type='svm', is_show=True, alpha=0.7, have_test=True)
    dimensions = classifier.dimensions  # dimensions is the number of the total features

    # call the instance of PSO
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(classifier.f, iters=10)

    classifier.time_print()

    if not n_fold:
        print(classifier.classify(pos))
        return pos,  classifier.classify(pos)
    else:
        # return pos,  classifier.classify(pos), classifier.data['train']
        return pos, classifier
         #  return pos,  classifier.classify(pos, is_show=True), classifier.data['train'], classifier.test_id


def n_fold_pso_svm(n=10, options='default', is_show=False, have_test=False):
    features = []
    timer_classify = Timer()
    for fold in range(n):
        # training pos_svm and predict

        with open('data/report.txt', 'a') as r:
            r.write('---------------------------Fold %d --------------------------------' % (fold + 1))
            r.write('\r\n')

        pos, model = pso_svm(options, n_fold=True, is_show=is_show)

        timer_classify.start()
        ac, mul_ac, res = model.classify(pos)
        feature_num = np.count_nonzero(np.array(pos))
        train_data = model.data['train']

        with open('data/report.txt', 'a') as r:
            r.write('training result -------------- accuracy: %.3f   feature_num：%d' % (ac, feature_num))
            r.write('\r\n')
            for cate_index in range(len(mul_ac)):
                r.write(Label_match[cate_index] + ' accuracy: %.3f' % (mul_ac[cate_index]))
                r.write('\r\n')

        if is_show:
            test_id = model.test_id
            gt = list(model.data['val'].iloc[:, 1])
            if len(res) == len(test_id):
                pass
            else:
                raise ValueError
            with open('data/report.txt', 'a') as r:
                r.write('------------optimized_prediction_list---------------')
                r.write('\r\n')
                r.write('exampleID      prediction      ground_truth')
                r.write('\r\n')
                for unit in range(len(test_id)):
                    r.write(test_id[unit] + '        ' + Label_match[res[unit]] + '          ' + Label_match[gt[unit]])
                    r.write('\r\n')

        if have_test:
            test_ac, test_res = model.classify(pos, have_test=True)
            test_gt = list(model.test_data[1].iloc[:, 1])
            example_id = model.final_test_id

            if len(test_res) == len(example_id) and len(test_res) == len(test_gt):
                pass
            else:
                raise ValueError

            with open('data/report.txt', 'a') as r:
                r.write('test result -------------------------- accuracy: %.3f' % (test_ac))
                r.write('\r\n')
                r.write('------------final_prediction_list---------------')
                r.write('\r\n')
                r.write('exampleID      prediction      ground_truth')
                r.write('\r\n')
                for unit in range(len(example_id)):
                    r.write(example_id[unit] + '        %.3f          %d' % (test_res[unit], test_gt[unit]))
                    r.write('\r\n')

        # turn pos into bool map
        pos = list(map(bool, pos))
        pos = [False, False] + pos
        # mask the bool map on the complete chart
        sub_set = train_data.columns[pos]
        features.append(sub_set)
        timer_classify.end()

    features = pd.DataFrame(features)
    features.to_csv('data/features_subset.csv')

    timer_classify.write_time("classification")


if __name__ == '__main__':
    timer_all = Timer()
    timer_all.start()
    n_fold_pso_svm(n=1, options='default', is_show=True, have_test=True)
    timer_all.end()
    timer_all.write_time("ALL")





