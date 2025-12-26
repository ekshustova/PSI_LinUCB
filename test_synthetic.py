import numpy as np
from scipy import linalg
import scipy.io
from tqdm import tqdm
import pickle as pkl

from models.dbsl import DBSL
from models.oful import OFUL
import argparse

from models.soful import SOFUL
from line_profiler import profile


@profile
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument("-l", type=int, help="Sketch size", required=False)
    parser.add_argument("-T", type=int, help="Time",
                        default=2000, required=False)
    parser.add_argument("-d", type=int, help="demision",
                        default=500, required=False)
    parser.add_argument("--lmd", type=float, help="Lambda", default=1.0, required=False)
    parser.add_argument("--beta", type=float, help="Beta", default=1e-4, required=False)
    parser.add_argument("--eps", type=float, help="Epsilon", required=False)
    args = parser.parse_args()

    method = args.method
    T = args.T
    d = args.d
    arms = 100
    beta = args.beta
    lmd = args.lmd

    theta = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    theta = theta / linalg.norm(theta)

    dataset = np.random.multivariate_normal(
        np.zeros(d), np.eye(d), (2000, arms)
    )
    row_norms = np.linalg.norm(dataset, axis=-1, keepdims=True)
    max_row_norm = np.max(row_norms)
    if max_row_norm != 0:
        dataset = np.divide(dataset, max_row_norm)
    row_norms = np.linalg.norm(dataset, axis=-1, keepdims=True) 
    max_row_norm = np.max(row_norms)
    acc_regret = 0.0
    acc_regrets = []
    l = args.l

    match method:
        case "oful":
            bandit = OFUL(d, beta=beta, lmd=lmd)
        case "soful":
            bandit = SOFUL(d, beta=beta, lmd=lmd, m=l)
        case "cbscfd":
            bandit = SOFUL(d, beta=beta, lmd=lmd, m=l, robust=True)
        case "dbsl":
            robust = True
            bandit = DBSL(d, l, args.eps, beta=beta, lmd=lmd, robust=robust)

    etas = np.random.normal(0, 1, arms)
    observe = lambda arm, x: x @ theta + etas[arm]

    for i in tqdm(range(T)):
        decision_set = dataset[i]
        reward = bandit.fit(decision_set, observe=observe)
        real_rewards = decision_set @ theta + etas
        best_real_reward = np.max(real_rewards)
        regret = best_real_reward - reward
        acc_regret += regret
        acc_regrets.append(acc_regret)

    X = bandit.X

if __name__ == "__main__":
    run()
