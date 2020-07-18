import sys
import experiment_ComplEx



def add_params(ps):
    args = []
    for key, value in ps.items():
        if key == 'quiet' or key == 'valid' or key == 'auc':
            args.append('--' + key)
        else:
            args.append('--' + key)
            args.append(str(value))

    #need to make this output the MRR scores
    final_metrics = experiment_ComplEx.main(args)

    return final_metrics
