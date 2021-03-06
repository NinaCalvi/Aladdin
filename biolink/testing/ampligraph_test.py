import numpy as np
from ampligraph.datasets import load_fb15k
from ampligraph.latent_features import ComplEx, TransE
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score

def main():

    # load Wordnet18 dataset:
    X = load_fb15k()

    # Initialize a ComplEx neural embedding model with pairwise loss function:
    # The model will be trained for 300 epochs.
    # model = ComplEx(batches_count=100, seed=3, epochs=1000, k=200, eta=10,
    #                 optimizer='adagrad', optimizer_params={'lr':0.5},
    #                 # Use pairwise loss with margin 0.5
    #                 loss='nll',
    #                 # Use L2 regularizer with regularizer weight 1e-5
    #                 regularizer='LP', regularizer_params={'p':2, 'lambda':0.01},
    #                 # Enable stdout messages (set to false if you don't want to display)
    #                 verbose=True)

    model = TransE(batches_count=100, seed=0, epochs=1000, k=50, eta=10,
                    optimizer='sgd', optimizer_params={'lr':0.01},
                    # Use pairwise loss with margin 0.5
                    loss='pairwise',
                    # Use L2 regularizer with regularizer weight 1e-5
                    regularizer='LP', regularizer_params={'p':2, 'lambda':0.01},
                    # Enable stdout messages (set to false if you don't want to display)
                    verbose=True)

    # For evaluation, we can use a filter which would be used to filter out
    # positives statements created by the corruption procedure.
    # Here we define the filter set by concatenating all the positives
    filter = np.concatenate((X['train'], X['valid'], X['test']))

    # Fit the model on training and validation set
    model.fit(X['train'],
              early_stopping = True,
              early_stopping_params = \
                      {
                          'x_valid': X['valid'],       # validation set
                          'criteria':'mrr',         # Uses hits10 criteria for early stopping
                          'burn_in': 100,              # early stopping kicks in after 100 epochs
                          'check_interval':50,         # validates every 20th epoch
                          'stop_interval':1,           # stops if 5 successive validation checks are bad.
                          'x_filter': filter,          # Use filter for filtering out positives
                          'corruption_entities':'all', # corrupt using all entities
                          'corrupt_side':'s+o'         # corrupt subject and object (but not at once)
                      }
              )



    # Run the evaluation procedure on the test set (with filtering).
    # To disable filtering: filter_triples=None
    # Usually, we corrupt subject and object sides separately and compute ranks
    ranks = evaluate_performance(X['test'],
                                 model=model,
                                 filter_triples=filter,
                                 use_default_protocol=False, # corrupt subj and obj separately while evaluating
                                 verbose=True)

    # compute and print metrics:
    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
    # Output: MRR: 0.886406, Hits@10: 0.935000

if __name__ == "__main__":
    main()
