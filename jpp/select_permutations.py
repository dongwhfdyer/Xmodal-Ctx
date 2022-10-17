# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:50:28 2017
It's used to select a collection of permutations.
And these permutations are tend to have a large distance between them.

@author: bbrattol
"""
import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Train network on Imagenet')
parser.add_argument('--classes', default=64, type=int,
                    help='Number of permutations to select')
parser.add_argument('--selection', default='max', type=str,
                    help='Sample selected per iteration based on hamming distance: [max] highest; [mean] average')
args = parser.parse_args()

if __name__ == "__main__":
    outname = 'permutations/permutations_hamming_%s_%d' % (args.selection, args.classes)

    P_hat = np.array(list(itertools.permutations(list(range(9)), 9)))  # shape: permutationsNum * 9
    n = P_hat.shape[0]  # number of permutations

    for i in trange(args.classes):  # choose the best args.classes permutation for each class
        # nonlocal P
        if i == 0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1, -1])  # randomly select one permutation
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)  # remove the selected permutation from the list
        D = cdist(P, P_hat, metric='hamming').mean(
            axis=0).flatten()  # compute the average hamming distance between the selected permutation and the remaining ones

        if args.selection == 'max':  # select the permutation with the highest average hamming distance
            j = D.argmax()
        else:  # select a permutation in the middle of the sorted list
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]

    np.save(outname, P)
    print('file created --> ' + outname)

    onePermutation = np.load(outname + '.npy')
    print("--------------------------------------------------")
