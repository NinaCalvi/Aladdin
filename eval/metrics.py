#contains possible different metrics

def rank(y_pred, true_idx):
    #sorts from smallest to biggest
    #we want biggest first (i.e. highest score)

    #applying argsort again will undo the sort done before
    #an assign to each element in the list its rank within the sorted stuff
    order_rank  = np.argsort(np.argsort(-y_pred))
    rank  = order_rank[true_idx] + 1
    return rank

def mean_rank(y_pred, true_idx):    
