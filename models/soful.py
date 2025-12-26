import numpy as np
import numpy.typing as npt
from functools import partial
import scipy.linalg as linalg
from typing import Callable
import utils
from frequent_directions import RobustFrequentDirections


class SOFUL:
    def __init__(
        self,
        d: int,
        beta: float,
        lmd: float,
        m: int,
        robust: bool = False,
        fast_fd: bool = True,
    ) -> None:
        self.time = 0
        self.d = d
        self.beta = beta
        self.lmd = lmd
        self.hat_theta = np.zeros((d, 1))
        self.X = np.zeros((0, d))

        self.m = m
        self.robust = robust
        self.fast_fd = fast_fd
        if not self.fast_fd:
            self.S = RobustFrequentDirections(d, m)
        else:
            self.S = np.zeros((m, d))
            self.H = (1 / lmd) * np.eye(m)

        self.deltas = np.array([self.lmd])

    def fit(
        self, decision_set: npt.NDArray, observe: Callable[[int, npt.ArrayLike], float]
    ) -> float:
        """
        decision_set: (num_actions, d)
        """
        self.time += 1

        if decision_set.ndim != 2:
            raise ValueError("Array dimension must be 2")

        S, H = self.S, self.H
        expected_rewards = (decision_set @ self.hat_theta).reshape(  # (10, 1)
            -1, 1
        ) + self.beta * (
            np.apply_along_axis(
                lambda x: np.sqrt(
                    np.abs(np.sum(np.square(x)) - (x @ S.T).T @ H @ (x @ S.T))
                    if len(S) != 0
                    else 0
                )
                / np.sqrt(self.lmd),
                1,
                decision_set,
            )  # (num_actions, 1)
        ).reshape(
            -1, 1
        )

        ind = np.argmax(expected_rewards)
        play = decision_set[ind]

        # observe reward
        reward = observe(ind, play)

        if not self.fast_fd:
            ## SOFUL: compute S_t, H_t using Alg. 4 given S_t-1, x_t
            self.S.fit(play)
            S, sigma_squared, _, delta = self.S.get()
            self.deltas.append(delta)
            H = 1 / (sigma_squared + self.lmd)
            self.V_inv = (np.eye(self.d) - S.T @ (H[:, np.newaxis] * S)) / self.lmd
            self.S = S
            self.H = H
        else:
            if len(self.S) >= 2 * self.m - 1:
                self.S = np.row_stack([self.S, play])
                _, s, Vt = linalg.svd(self.S, full_matrices=False, overwrite_a=True)
                sigma_squared = s**2
                if len(sigma_squared) > self.m:
                    delta = sigma_squared[self.m]
                    if self.robust:
                        self.lmd += delta
                    self.deltas = np.row_stack([self.deltas, delta])
                    sigma_squared = sigma_squared[: self.m] - sigma_squared[self.m]
                    Vt = Vt[: self.m]
                self.S = Vt * np.sqrt(sigma_squared).reshape(-1, 1)
                S = self.S
                H = 1 / (sigma_squared + self.lmd)
                H = np.diag(H)
                self.H = H
            else:
                p = self.H @ self.S @ play
                k = (
                    np.sum(np.square(play))
                    - play.reshape(1, -1) @ self.S.T @ p
                    + self.lmd
                )
                self.H = np.block(
                    [
                        [self.H + np.outer(p, p) / k, -p.reshape(-1, 1) / k],
                        [-p.reshape(1, -1) / k, 1 / k],
                    ]
                )
                self.S = np.row_stack([self.S, play])
                S = self.S
                H = self.H
                self.deltas = np.row_stack([self.deltas, 0])

        # compute the V_inv and theta
        self.X = np.row_stack([self.X, play])

        self.xy += reward * play

        self.hat_theta = self.xy.T - S.T @ (H @ (S @ self.xy.T))
        self.hat_theta /= self.lmd

        return reward
