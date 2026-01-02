import numpy as np
import numpy.typing as npt
from functools import partial
import scipy.linalg as linalg
from typing import Callable
import utils
from frequent_directions import DyadicBlockSketching
from line_profiler import profile


class DBSL:
    def __init__(
        self,
        d: int,
        sketch_size: int,
        eps: float,
        beta: float,
        lmd: float,
        robust: bool = True,
    ) -> None:
        self.time = 0
        self.d = d
        self.beta = beta
        self.lmd = lmd
        self.hat_theta = np.zeros((d, 1))
        self.X = np.zeros((0, d))
        self.xy = np.zeros((1, d))

        self.sketch_size = sketch_size
        self.robust = robust
        self.sketch = DyadicBlockSketching(sketch_size, d, eps, lmd, robust)

        self.deltas = np.array([self.lmd])

    #@profile
    def fit(
        self, decision_set: npt.NDArray, observe: Callable[[int, npt.ArrayLike], float]
    ) -> float:
        """
        decision_set: (num_actions, d)
        """
        S, H = self.sketch.get()
        self.time += 1

        if decision_set.ndim != 2:
            raise ValueError("Array dimension must be 2")

        expected_rewards = (decision_set @ self.hat_theta).reshape(
            -1, 1
        ) + self.beta * (
            np.apply_along_axis(
                lambda x: np.sqrt(
                    np.sum(np.square(x)) - (x @ S.T).T @ H @ (x @ S.T)
                    if len(S) != 0
                    else 0
                )
                / np.sqrt(self.lmd),
                1,
                decision_set,
            )
        ).reshape(
            -1, 1
        )

        ind = np.argmax(expected_rewards)
        play = decision_set[ind] #это видимо контекст

        # observe reward
        reward = observe(ind, play)

        ## SOFUL: compute S_t, H_t using Alg. 4 given S_t-1, x_t
        self.sketch.fit(play) #elf.sketchэто гет скетч
        S, H = self.sketch.get()
        self.hat_theta = self.hat_theta
        if len(S) != 0:
            self.xy += reward * play
            self.hat_theta = self.xy.T - S.T @ (H @ (S @ self.xy.T)) #A-1 rx
            self.hat_theta /= self.lmd

        # compute the V_inv and theta
        self.X = np.row_stack([self.X, play])
        self.xy += reward * play

        return reward


def demo_call_fit_dbsl():
    d = 5
    num_actions = 10

    model = DBSL(
        d=d,
        sketch_size=20,
        eps=5.0,
        beta=1.0,
        lmd=1.0,
        robust=True,
    )

    # (num_actions, d)
    decision_set = np.random.randn(num_actions, d)


    true_theta = np.random.randn(d)

    def observe(ind, x):
        return float(x @ true_theta + 0.1 * np.random.randn())

    r = model.fit(decision_set, observe)
    print("reward:", r)


if __name__ == "__main__":
    demo_call_fit_dbsl()
