import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == '__main__':

    inscountfile = './localdata/cate2insnum.pkl'
    with open(inscountfile, 'rb') as f:
        inscount = pickle.load(f)

    paramfile = './localdata/r50_param_ana.pkl'
    with open(paramfile, 'rb') as f:
        param = pickle.load(f)

    x = np.arange(1231)
    y1 = inscount['train']
    y2 = inscount['val']
    # y1[0] = 100000000000
    y3 = param[0] # weight_norm
    y4 = param[1] # bias

    sort_idx = np.argsort(-y3)

    yy1 = y1[sort_idx]
    yy2 = y2[sort_idx]
    yy3 = y3[sort_idx]
    yy4 = y4[sort_idx]

    plt.figure()
    # l1, = plt.plot(x, yy1, color='red', linestyle='-', label='train_ins')
    # l2, = plt.plot(x, yy2, color='blue', linestyle='-', label='val_ins')
    l3, = plt.plot(x, yy3, color='green', linestyle='-', label='weight')
    # l4, = plt.plot(x, yy4, color='cyan', linestyle='-', label='bias')

    plt.legend(loc='upper right')

    plt.show()