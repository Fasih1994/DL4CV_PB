import numpy as np


def ranked5_accuracy(preds, labels):
    # initialize rank1 an rank5 accuracy
    ranked1 = 0
    ranked5 = 0
    for (p, gt) in zip(preds, labels):
        # sort the probabilities by their index in descending
        # order so that the more confident guesses are at the
        # front of the list
        p = np.argsort(p)[::-1]

        # check if gt in top 5 predictions
        if gt in p[:5]:
            ranked5 += 1
        # check if gt is top prediction
        if gt == p[0]:
            ranked1 += 1

    ranked1 /= float(len(labels))
    ranked5 /= float(len(labels))

    return (ranked1, ranked5)
