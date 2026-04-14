import os
import argparse
import numpy as np
from pathlib import Path
from Utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_filepath", type=str)
    parser.add_argument("label_filepath", type=str)
    arg = parser.parse_args()

    num_class = 10
    num_img, n_r, n_c, imgs, num_label, labels = \
        parse_dataset(arg.img_filepath, arg.label_filepath)
    thres = 128
    data = (imgs > thres).astype(int)
    # init
    P = np.random.rand(num_class, n_r, n_c)
    lam = np.random.rand(num_class)
    lam = lam / lam.sum()
    gamma = np.zeros((num_img, num_class))
    epoch = 1000
    for _ in range(epoch):
        # E step
        for i, d in enumerate(data):
            prob = np.ones(num_class)
            for r in range(n_r):
                for c in range(n_c):
                    prob *=  P[:, r, c]**(d[r, c]) * \
                        (1 - P[:, r, c])**(1 - d[r, c])
            gamma[i] = lam * prob
            gamma[i] = gamma[i] / gamma[i].sum()
        # M step
        sum_gamma = gamma.sum(axis=0)
        lam = sum_gamma / num_img
        for r in range(n_r):
            for c in range(n_c):
                sum_gamma_data = np.zeros(num_class)
                for i, d in enumerate(data):
                    sum_gamma_data += gamma[i] * d[r, c]
                P[:, r, c] = sum_gamma_data / sum_gamma
