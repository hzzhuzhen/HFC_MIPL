import numpy as np
import scipy.io as sio
from metric import metric
from sklearn import preprocessing
import math
from sklearn.metrics import f1_score, label_ranking_loss, hamming_loss, coverage_error, average_precision_score


def convert_labels(Y):
    Y = (Y > 0) * 1
    return Y
def load_data(path, tr_rate=0.8, observe_rate=0.2):
    data = sio.loadmat(path)

    if path.find('nus') >= 0:
        X = data['data'].T
        X = preprocessing.scale(X)
        Y = data['target'].T
        Y_P = data['train_p_target'].T
    else:
        X = data['data']
        X = preprocessing.scale(X)
        # pre-processing
        Y = data['target'].T
        Y_P = data['partial_labels'].T
    print('Data size (X, Y, Y_P): ', X.shape, Y.shape, Y_P.shape)

    data_num = int(X.shape[0])
    perm = np.arange(data_num)
    np.random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]
    Y_P = Y_P[perm]
    part = int(data_num * tr_rate)
    trX = X[0:part, :]
    trY = Y[0:part, :]
    trpY = Y_P[0:part, :]
    tsX = X[part:data_num, :]
    tsY = Y[part:data_num, :]

    return (trX, trY, trpY, tsX, tsY)


def load_data_k_fold(path, k=10):
    data = sio.loadmat(path)
    if path.find('nus') >= 0:
        X = data['data'].T # transpose X for NUSWIDE
        X = preprocessing.scale(X)
        Y = data['target'].T
        Y_P = data['train_p_target'].T
    else:
        X = data['data']
        X = preprocessing.scale(X)
        # pre-processing
        Y = data['target'].T
        Y_P = data['partial_labels'].T

    data_num = int(X.shape[0])
    perm = np.arange(data_num)
    np.random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]
    Y_P = Y_P[perm]

    X_folds = [None] * k
    Y_folds = [None] * k
    YP_folds = [None] * k
    for i in range(k):
        start = int(data_num * (1.0 * i / k))
        if i < k - 1:
            end = int(data_num * (1.0 * (i+1) / k))
        else:
            end = data_num
        X_folds[i] = X[start:end, :]
        Y_folds[i] = Y[start:end, :]
        YP_folds[i] = Y_P[start:end, :]

    return (X_folds, Y_folds, YP_folds)

def evaluate(Y_test, scores, threshold=0.5):
    return metric(Y_test, 1 * (scores > threshold), scores)


def load_preprocessed_data(path):
    data = sio.loadmat(path + '_preprocessed')
    return (data['trX'].copy(), data['trY'].copy(), data['trpY'].copy(), data['tsX'].copy(), data['tsY'].copy())
if __name__ == '__main__':
    path = 'data/music_emotion'
    trX, trY, trpY, tsX, tsY = load_data(path)
    sio.savemat(path + '_preprocessed.mat', {
        'trX': trX,
        'trY': trY,
        'trpY': trpY,
        'tsX': tsX,
        'tsY': tsY
    })
    pass