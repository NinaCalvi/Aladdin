#contain useful functions for different needs
import torch

def generate_neg_instances(triples: torch.Tensor, nb_corrs: int, nb_ents: int, seed: int, *args, **kwargs):
    '''
    Generate random negatives for some positive triples
    This is needed when we are using pairwise/pointwise losses
    i.e. NOT needed for mcl models

    Parameters:
    -----------
    triples: positive triples with size [?, 3]
    nb_corss: number of corruptions to generate per triple
    nb_ents: total number of entities
    seed: random seed

    Returns:
    ---------
    torch.Tensor
        torch tensor for negative triples of size [?, 3]

    Note
    ---------
    The passed `nb_corrs` is evenly distributed between head and tail corruptions.
    i.e. create equal number of corruption on the subject and on the object

    Warning
    ---------
    This corruption heuristic might generate original true triples as corruptions.
    '''

    #create split
    nb_corrs /= 2
    obj_corruptions = triples.repeat(ceil(nb_corrs), 1)
    subj_corruptions = triples.repeat(florr(nb_corrs), 1)

    #split tensors to isolate subject or object for corruption purposes
    sub_pred, obj = torch.split(obj_corruptions, [2, 1], dim=1)
    sub, obj_pred = torch.split(subj_corruptions, [1,2], dim=1)

    #corrupt
    obj.random_(0, nb_ents, seed=seed)
    sub.random_(0, nb_ents, seed=seed)

    return torch.cat((torch.cat((sub_pred, obj), axis=1), torch.cat((sub ,obj_pred), axis=1)), axis=0)
