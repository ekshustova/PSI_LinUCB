import time
#from datasetsmnist import MNIST
import pandas as pd
import numpy as np
import argparse

from tqdm import trange

from models.dbsl import DBSL

from models.oful import OFUL
from models.soful import SOFUL

import pickle as pkl
from scipy import stats
from line_profiler import profile
from datetime import datetime


def groupby(features, labels):
    df = pd.DataFrame(np.hstack([labels[:, np.newaxis], features]))

    grouped = df.groupby(0)

    labels = []
    clusters = {}

    for label, group in grouped:
        labels.append(int(label))
        clusters[int(label)] = group.values[:, 1:]

    return labels, clusters


# @profile
def run():
    np.seterr("raise")
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument("-l", type=int, help="Sketch size", required=False)
    parser.add_argument("--eps", type=float, help="Epsilon", required=False)
    parser.add_argument("--lmd", type=float, help="Lambda", required=True)
    parser.add_argument("--beta", type=float, help="Beta", required=True)
    args = parser.parse_args()

    method = args.method
    eps = args.eps

    mndata = MNIST("datasets/MNIST/raw")
    images, labels = mndata.load_training()
    features = np.array(images)
    row_norms = np.linalg.norm(features, axis=1, keepdims=True)
    max_row_norm = np.max(row_norms)
    print(max_row_norm)
    if max_row_norm != 0:
        features = np.divide(features, max_row_norm)
    row_norms = np.linalg.norm(features, axis=1, keepdims=True) 
    max_row_norm = np.max(row_norms)
    print(max_row_norm)

    labels = np.array(labels)

    labels, clusters = groupby(features, labels)

    arms = len(labels)
    d = features.shape[-1]
    l = args.l
    beta = args.beta
    lmd = args.lmd

    T = 2000

    acc_regret = 0.0
    acc_regrets = []

    match method:
        case "oful":
            bandit = OFUL(d, beta=beta, lmd=lmd)
        case "soful":
            bandit = SOFUL(d, beta=beta, lmd=lmd, m=l)
        case "cbscfd":
            bandit = SOFUL(d, beta=beta, lmd=lmd, m=l, robust=True)
        case "dbsl":
            robust = True
            bandit = DBSL(d, l, eps, beta, lmd, robust=robust)

    observe = lambda arm, x: 1 if arm == 0 else 0

    start_time = time.process_time_ns()
    for i in trange(T):
        decision_set = np.zeros((arms, d))
        for key in clusters:
            row_num = clusters[key].shape[0]
            decision_index = np.random.choice(row_num)
            decision_set[key] = clusters[key][decision_index]

        reward = bandit.fit(decision_set, observe=observe)
        best_rewards = 1
        regret = best_rewards - reward
        acc_regret += regret
        acc_regrets.append(acc_regret)

    X = bandit.X

if __name__ == "__main__":
    run()
