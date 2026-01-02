# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Self
from line_profiler import profile
import numpy as np
import numpy.typing as npt
from scipy import linalg
import utils

SVD_COUNT_OURS = 0
FLUSH_HIT = 0
FLUSH_ENTER = 0


class FrequentDirections:
    def __init__(self, d, sketch_dim=8):
        """
        Class wrapper for all FD-type methods

        __rotate_and_reduce__ is not defined for the standard FrequentDirections but is for the
        subsequent subclasses which inherit from FrequentDirections.
        """
        self.d = d
        self.delta = 0.0  # For RFD

        if sketch_dim is not None:
            self.sketch_dim = sketch_dim
        self.sketch = np.zeros((self.sketch_dim, self.d), dtype=float)
        self.Vt = np.zeros((self.sketch_dim, self.d), dtype=float)
        self.sigma_squared = np.zeros(self.sketch_dim, dtype=float)

        self.svd_count = 0

    # @profile
    def fit(self, X: npt.NDArray, batch_size=1):
        """
        Fits the FD transform to dataset X
        """
        global SVD_COUNT_OURS
        if X.ndim == 1:
            X = X[np.newaxis, :]
        n = X.shape[0]
        for i in range(0, n, batch_size):
            aux = np.zeros((self.sketch_dim + batch_size, self.d))
            batch = X[i : i + batch_size, :]
            # aux = np.concatenate((self.sketch, batch), axis=0)
            aux[0 : self.sketch_dim, :] = self.sketch
            aux[self.sketch_dim : self.sketch_dim + batch.shape[0], :] = batch
            # ! WARNING - SCIPY SEEMS MORE ROBUST THAN NUMPY SO COMMENTING THIS WHICH IS FASTER OVERALL
            # try:
            #     _, s, self.Vt = np.linalg.svd(aux, full_matrices=False)
            # except np.linalg.LinAlgError:
            #     _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesvd')
            # print(aux.shape)
            _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver="gesvd")

            # self.svd_count += 1
            SVD_COUNT_OURS += 1

            self.sigma_squared = s**2
            self.__rotate_and_reduce__()
            self.sketch = self.Vt * np.sqrt(self.sigma_squared).reshape(-1, 1)

    def get(self):
        return self.sketch, self.sigma_squared, self.Vt, self.delta

    def get_sketch(self):
        return self.sketch

    def __rotate_and_reduce__(self):
        if len(self.sigma_squared) > self.sketch_dim:
            self.delta += self.sigma_squared[self.sketch_dim]
            self.sigma_squared = (
                self.sigma_squared[: self.sketch_dim]
                - self.sigma_squared[self.sketch_dim]
            )
            self.Vt = self.Vt[: self.sketch_dim]


class RobustFrequentDirections(FrequentDirections):
    """
    Implements the RFD version of FD by maintaining counter self.delta.
    Still operates in the `fast` regimen by doubling space, as in
    FastFrequentDirections
    """

    def __rotate_and_reduce__(self):
        if len(self.sigma_squared) > self.sketch_dim:
            self.delta += self.sigma_squared[self.sketch_dim]
        super().__rotate_and_reduce__()


class FastFrequentDirections(RobustFrequentDirections):
    """
    Implements the fast version of FD by doubling space
    """

    def __init__(self, d: int, sketch_dim: int):
        super().__init__(d, min(sketch_dim, d))
        self.buffer = None

    def __flush(self):
        if self.buffer is not None:
            super().fit(
                self.buffer, batch_size=min(self.buffer.shape[0], self.sketch_dim)
            )
            self.buffer = None

    def fit(self, X):
        if self.buffer is None:
            self.buffer = X
        else:
            self.buffer = np.concatenate([self.buffer, X])

        if self.buffer is not None and len(self.buffer) >= self.sketch_dim:
            self.__flush()

    def get(self):
        self.__flush()
        return super().get()

    # def __rotate_and_reduce__(self):
    #     self.sigma_squared = (
    #         self.sigma_squared[: self.sketch_dim] - self.sigma_squared[self.sketch_dim]
    #     )
    #     self.Vt = self.Vt[: self.sketch_dim]


class RegularizedFD:
    def __init__(self, sketch_size: int, d: int, lmd: float, robust: bool = True):
        self.sketch_size = sketch_size
        self.d = d
        self.lmd = lmd
        self.robust = robust

        self.S = np.zeros((0, self.d))
        self.H = np.zeros((0, 0))

    def fit(self, X: npt.NDArray) -> bool:
        if len(self.S) >= 2 * self.sketch_size - 1:
            self.S = np.row_stack([self.S, X])
            _, s, Vt = linalg.svd(self.S, full_matrices=False, overwrite_a=True)
            sigma_squared = s**2
            if len(sigma_squared) > self.sketch_size:
                delta = sigma_squared[self.sketch_size]
                if self.robust:
                    self.lmd += delta
                sigma_squared = (
                    sigma_squared[: self.sketch_size] - sigma_squared[self.sketch_size]
                )
                Vt = Vt[: self.sketch_size]
            self.S = Vt * np.sqrt(sigma_squared).reshape(-1, 1)
            H = 1 / (sigma_squared + self.lmd)
            H = np.diag(H)
            self.H = H
            return True
        else:
            p = self.H @ (self.S @ X)
            k = np.sum(np.square(X)) - X.reshape(1, -1) @ self.S.T @ p + self.lmd
            self.H = np.block(
                [
                    [self.H + np.outer(p, p) / k, -p.reshape(-1, 1) / k],
                    [-p.reshape(1, -1) / k, 1 / k],
                ]
            )
            self.S = np.row_stack([self.S, X])
            return False

    def get(self):
        return self.S, self.H

    def __iadd__(self, other: Self) -> Self:
        self.lmd += other.lmd
        if self.S.shape[0] == 0:
            self = other
            return self

        S, H = self.get()
        S_B, H_B = other.get()
        self.S = np.vstack([S, S_B])
        self.H = self.S @ self.S.T
        self.H += self.lmd * np.eye(self.H.shape[0])
        self.H = linalg.inv(self.H)
        return self

    def __add__(self, other: Self) -> Self:
        ret = deepcopy(self)
        ret += other
        return ret

    def size(self):
        return self.S.size + self.H.size


@dataclass
class DyadicBlock:
    sketch: RegularizedFD
    size: float = 0.0
    length: int = 0
    error: float = 0.0

    def __iadd__(self, block: Self) -> Self:
        self.sketch += block.sketch
        self.size += block.size
        self.length += block.length
        if 0 < block.sketch.sketch_size <= self.sketch.d:
            self.error += block.size / block.length
        return self

    def __add__(self, other: Self) -> Self:
        ret = deepcopy(self)
        ret += other
        return ret

    def get_size(self):
        return self.sketch.size()


class DyadicBlockSketching:
    def __init__(
        self, sketch_size: int, d: int, eps: float, lmd: float, robust: bool = True, C: int = 2
    ):
        self.sketch_size = sketch_size
        self.d = d
        self.eps = eps
        self.lmd = lmd
        self.robust = robust

        self.inactive_block = DyadicBlock(
            sketch=RegularizedFD(0, self.d, self.lmd, self.robust)
        )
        self.active_block = DyadicBlock(
            sketch=RegularizedFD(self.sketch_size, self.d, self.lmd, self.robust),
            length=self.sketch_size,
        )
        self.S = np.zeros((0, d))
        self.H = np.zeros((0, 0))

        self.size_threshold = self.eps * self.sketch_size / 2

        self.C: int = C

    def fit(self, X: npt.NDArray):
        size = np.inner(X, X)
        if (
            self.active_block.size + size > self.size_threshold
            and self.active_block.length < int(self.C * (self.d + 1))
        ):
            # merge
            self.inactive_block += self.active_block
            length = min(int(self.C * self.active_block.length), int(self.C * (self.d + 1)))

            self.active_block = DyadicBlock(
                sketch=RegularizedFD(
                    sketch_size=length, d=self.d, lmd=self.lmd, robust=self.robust
                ),
                length=length,
            )

        reduced = self.active_block.sketch.fit(X)
        if reduced:
            sketch = self.inactive_block.sketch + self.active_block.sketch
            self.S, self.H = sketch.get()
        else:
            p = self.H @ (self.S @ X)
            k = np.sum(np.square(X)) - X.reshape(1, -1) @ self.S.T @ p + self.lmd
            self.H = np.block(
                [
                    [self.H + np.outer(p, p) / k, -p.reshape(-1, 1) / k],
                    [-p.reshape(1, -1) / k, 1 / k],
                ]
            )
            self.S = np.row_stack([self.S, X])

        self.active_block.size += size

    def get(self):
        return self.S, self.H
