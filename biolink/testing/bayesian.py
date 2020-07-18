import sys
import experiment_ComplEx

import numpy as np
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials, fmin
from hyperopt.pyll import scope
import csv

output_file = None
iteration = 0




space = {'model': hp.choice('model', ['tucker']), \
        'mcl': hp.choice('mcl', [True]), \
        'data': hp.choice('data', ['pse']), \
        'learning-rate': hp.uniform('learning-rate', 0.0003, 1), \
        'batch-size': hp.choice('batch-size', [128, 268, 512]), \
        'reg-weight': hp.loguniform('reg-weight', np.log(1.0e-20), np.log(1.0e-01)), \
        'embedding-size': scope.int(hp.quniform('embedding-size', 50, 200,1)), \
        'rel-emb-size': scope.int(hp.quniform('real-emb-size', 50, 200,1)), \
        'quiet': hp.choice('quiet', [True]), \
        'valid': hp.choice('valid', [True])}



def do_hyperopt(parameter_space, num_eval):
    trials = Trials()
    bp = fmin(add_params, parameter_space, algo=tpe.suggest, max_evals=2, trials=trials)
    print(bp)
    return trials

def add_params(ps):
    global iteration
    iteration += 1

    args = []
    for key, value in ps.items():
        if key == 'quiet' or key == 'valid' or key == 'auc':
            args.append('--' + key)
        else:
            args.append('--' + key)
            args.append(str(value))

    #need to make this output the MRR scores
    print(args)
    metrics = experiment_ComplEx.main(args, bayesian=True)

    with open(output_file, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(-metrics['MRR'], metrics['H@1'], metrics['H@3'], metrics['H@10'], \
        iteration, args)

    #taking negative since we acutally want to maximise mrr
    return {'loss': -metrics['MRR'], 'H@1': metrics['H@1'], 'H@3': metrics['H@3'], \
            'H@10': metrics['H@10'], 'status': STATUS_OK}
    # return final_metrics

def main(name_file):
    global output_file
    output_file = name_file + '.csv'

    of_connection = open(output_file, 'w')
    writer = csv.writer(of_connection)
    writer.writerow(['MRR', 'h@1', 'h@3', 'h@10', 'iteration', 'params'])
    of_connection.close()


    num_eval = 50
    # trials = Trials()
    trials = do_hyperopt(space, num_eval)
    mrr = -1 * trials.best_trial['result']['loss']
    h1 = trials.best_trial['result']['H@1']
    h3 = trials.best_trial['result']['H@3']
    h10 = trials.best_trial['result']['H@10']

    print('------------- BEST PARAMS ---------')

    for key, val in space.items():
        print('\n')
        print(key, trials.best_trial['misc']['vals'][key])
    print('------------- BEST RESULTS ---------')
    print(f'MRR {mrr}, H@1 {h1}, H@3 {h3}, H@10 {h10}')



if __name__ == '__main__':
    main(sys.argv[1])
