import numpy as np
import numpy.typing as npt
from functools import partial
from typing import Callable
import utils
from line_profiler import profile


class OFUL:
    def __init__(self, d: int, beta: float, lmd: float) -> None:
        self.time = 0
        self.d = d
        self.beta = beta
        self.lmd = lmd
        self.hat_theta = np.zeros((d, 1))
        self.V_inv = (1 / lmd) * np.eye(d)
        self.X = np.zeros((0, d))
        self.rewards = np.zeros((0))

    @profile
    def fit(
        self, decision_set: npt.NDArray, observe: Callable[[int, npt.ArrayLike], float]
    ) -> float:
        """
        decision_set: (num_actions, d)
        """
        self.time += 1

        if decision_set.ndim != 2:
            raise ValueError("Array dimension must be 2")

        expected_rewards = (
            decision_set @ self.hat_theta
            + self.beta
            * np.apply_along_axis(
                partial(utils.matrix_induced_norm, A=self.V_inv), 1, decision_set
            )
        )  # (num_actions, 1)

        ind = np.argmax(expected_rewards)
        play = decision_set[ind]

        # observe reward
        reward = observe(ind, play)

        # compute the V_inv and theta
        self.X = np.row_stack([self.X, play])
        self.rewards = np.append(self.rewards, reward)

        self.V_inv = utils.woodbury(self.V_inv, play, play)
        self.hat_theta = self.V_inv @ (self.rewards @ self.X)

        return reward
