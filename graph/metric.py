import numpy as np
from sklearn.metrics import label_ranking_loss, hamming_loss, average_precision_score
def metric(y, y_pred, scores):
    n = 1.0 * y.shape[0]
    r_loss = label_ranking_loss(y, scores)
    h_loss = hamming_loss(y, y_pred)
    ap = 0
    for i in range(y.shape[0]):
        ap += average_precision_score(y[i], scores[i])

    ap /= n
    return r_loss, h_loss, ap

if __name__ == '__main__':
    pass