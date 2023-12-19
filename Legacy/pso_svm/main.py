import math
from Classifier import Classifier
import pyswarms as ps
import pandas as pd
import numpy as np
from Timer import Timer

Label_match = {0: 'Health', 1: 'cancer in stomach', 2: 'cancer in colon', 3: 'cancer in kidney',
               4: 'cancer in pancreas', 5: 'cancer in lung', 6: 'cancer in breast'}


def pso_svm(options='default'):
    # Initialize the swarm, arbitrary
    if options == 'default':
        options = {'c1': 0.5 + math.log(2), 'c2': 0.5 + math.log(2), 'w': 1/(2 * math.log(2)), 'k': 3, 'p': 2}
    # create the classification instance
    classifier = Classifier(alpha=0.7)
    dimensions = classifier.dimensions  # dimensions is the number of the total features

    # call the instance of PSO
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(classifier.f, iters=1)

    classifier.time_print()

    return pos, classifier


def n_fold_pso_svm(n=10, options='default', is_show=False, have_test=False):
    features = []
    for fold in range(n):
        # training pos_svm and predict
        with open('data/report.txt', 'a') as r:
            r.write('---------------------------Fold %d --------------------------------' % (fold + 1))
            r.write('\r\n')
        # pso swarm optimization
        pos, model = pso_svm(options)

        # ac, mul_ac, res = model.classify(pos, mode="train")
        # feature_num = np.count_nonzero(np.array(pos))
        # train_data = model.data['train']
        #
        # with open('data/report.txt', 'a') as r:
        #     r.write('training result -------------- accuracy: %.3f   feature_num：%d' % (ac, feature_num))
        #     r.write('\r\n')
        #     for cate_index in range(len(mul_ac)):
        #         r.write(Label_match[cate_index] + ' accuracy: %.3f' % (mul_ac[cate_index]))
        #         r.write('\r\n')

        total_ac, ac, res = model.classify(pos, mode="test")
        print('test result -------------------------- accuracy: %.3f' % (total_ac))
        print(ac)
        print(res)
        # with open('data/report.txt', 'a') as r:
        #     r.write('test result -------------------------- accuracy: %.3f' % (test_ac))
        #     r.write('\r\n')
        #     r.write('------------final_prediction_list---------------')
        #     r.write('\r\n')
        #     r.write('exampleID      prediction      ground_truth')
        #     r.write('\r\n')
        #     for unit in range(len(example_id)):
        #         r.write(example_id[unit] + '        %.3f          %d' % (test_res[unit], test_gt[unit]))
        #         r.write('\r\n')

        # turn pos into bool map
        pos = list(map(bool, pos))
        pos = [False] + pos
        # mask the bool map on the complete chart
        sub_set = model.data['train'].columns[pos]
        features.append(sub_set)

    features = pd.DataFrame(features)
    features.to_csv('data/features_subset.csv')



if __name__ == '__main__':
    timer_all = Timer()
    timer_all.start()
    n_fold_pso_svm(n=1, options='default', is_show=True, have_test=True)
    timer_all.end()
    timer_all.write_time("ALL")





