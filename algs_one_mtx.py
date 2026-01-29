import numpy as np
from numpy.linalg import svd
from collections import defaultdict
from numpy.linalg import multi_dot
from scipy.linalg import qr, norm
from math import sqrt


class CBSCFD:
    def __init__(self, lambd, beta, m, d):
        self.d = d
        self.beta = beta
        self.alpha = lambd
        self.m = m

        self.theta = np.zeros(self.d)
        self.Z = np.zeros((0, self.d))
        self.H = None
        self.H_prev = None
        self.sum_xy = np.zeros(self.d)
        self.t = 0

    def _apply_V_inv(self, x):
        if self.Z.shape[0] == 0:
            return x / self.alpha
        Zx = self.Z @ x
        HZx = self.H @ Zx
        ZT_HZx = self.Z.T @ HZx
        return (x - ZT_HZx) / self.alpha

    def update(self, context, r):
        self.t += 1
        x_t = context
        self.sum_xy += x_t * r
        n_prev = self.Z.shape[0]
        self.Z = np.vstack([self.Z, x_t.reshape(1, -1)])
        n = self.Z.shape[0]

        if n == 2 * self.m:
            self._svd_update()
        else:
            self._sequential_update(x_t, n_prev)

        self.theta = self._apply_V_inv(self.sum_xy)

    def _svd_update(self):
        U, Sigma, Vt = svd(self.Z, full_matrices=False)
        if len(Sigma) >= self.m:
            delta_t = Sigma[self.m - 1] ** 2
        else:
            delta_t = 0
        self.alpha += delta_t
        Sigma_hat_sq = np.maximum(Sigma[:self.m] ** 2 - delta_t, 0)
        Sigma_hat = np.sqrt(Sigma_hat_sq)
        # nonzero = Sigma_hat > 1e-10
        # self.Z = Sigma_hat[nonzero].reshape(-1, 1) * Vt[:self.m][nonzero]
        self.Z = Sigma_hat.reshape(-1, 1) * Vt[:self.m, :]
        H_diag = 1.0 / (Sigma_hat_sq + self.alpha)
        self.H = np.diag(H_diag)
        self.H_prev = self.H.copy()

    def _sequential_update(self, x_t, n_prev):
        n = self.Z.shape[0]
        if n == 1:
            k = np.dot(x_t, x_t) + self.alpha
            self.H = np.array([[1.0 / k]])
            self.H_prev = self.H
        else:
            if self.H_prev is None or self.H_prev.shape[0] != n_prev:
                self.H_prev = np.eye(n_prev) / self.alpha
            Z_prev = self.Z[:n_prev, :]
            Z_prev_x = Z_prev @ x_t
            p = self.H_prev @ Z_prev_x
            k_scalar = np.dot(x_t, x_t) - np.dot(Z_prev_x, p) + self.alpha
            self.H = np.zeros((n, n))
            self.H[:n - 1, :n - 1] = self.H_prev + np.outer(p, p) / k_scalar
            self.H[:n - 1, n - 1] = -p / k_scalar
            self.H[n - 1, :n - 1] = -p / k_scalar
            self.H[n - 1, n - 1] = 1.0 / k_scalar
            self.H_prev = self.H

    def score(self, context):
        x = context
        mean = np.dot(self.theta, x)
        var = np.dot(x, self._apply_V_inv(x))
        return mean + self.beta * np.sqrt(max(var, 0))

    def select_arm(self, contexts):
        scores = [self.score(ctx) for ctx in contexts]
        return int(np.argmax(scores))


def integrator(tilde_Ut, S, tilde_Vt, delta_U, delta_V):
    Ut = tilde_Ut.copy()
    Vt = tilde_Vt.copy()
    K1 = Ut @ S + delta_U.dot(delta_V.T.dot(Vt))
    tilde_U1, tilde_S1 = qr(K1, mode="economic")
    tilde_S0 = tilde_S1 - tilde_U1.T.dot(delta_U.dot(delta_V.T.dot(Vt)))
    L1 = Vt.dot(tilde_S0.T) + delta_V.dot(delta_U.T.dot(tilde_U1))
    tilde_V1, S1 = qr(L1, mode="economic")
    S1 = S1.T
    return tilde_U1, S1, tilde_V1


def svd_U_V_T(U, V, rank):
    Q_U, R_U = qr(U, mode="economic")
    Q_V, R_V = qr(V, mode="economic")

    U_svd, S_svd, V_svd = svd(R_U.dot(R_V.T), full_matrices=False)

    U_svd = U_svd[:, :rank]
    S_svd = S_svd[:rank]
    V_svd = V_svd[:rank, :]

    U_new = Q_U.dot(U_svd)
    V_new = V_svd.dot(Q_V.T)

    return U_new, np.diag(S_svd), V_new.T





class LinUCB_SM:
    def __init__(self, d, alpha, epsilon=1.0):
        self.d = d
        self.alpha = alpha
        self.epsilon = epsilon

        self.A_inv = np.eye(self.d) / self.epsilon
        self.b = np.zeros(self.d)
        self.theta = np.zeros(self.d, dtype=np.float32)
        self.update_count = 0

    def update(self, context, r):
        x_ua = context
        self.b += r * x_ua

        A_inv_x = self.A_inv @ x_ua
        self.A_inv -= np.outer(A_inv_x, A_inv_x) / (1 + x_ua.T @ A_inv_x)

        self.theta = self.A_inv @ self.b
        self.update_count += 1

    def score(self, context):
        ctx = context
        mean = float(np.dot(self.theta.T, ctx))
        exp = np.sqrt(np.dot(ctx.T, self.A_inv @ ctx))
        return mean + self.alpha * exp

    def select_arm(self, contexts):
        scores = [self.score(ctx) for ctx in contexts]
        return int(np.argmax(scores))


class CBRAP:
    def __init__(self, d, lambd=1.0, beta=1.0, m=50):
        self.d = d
        self.m = m
        self.alpha = lambd
        self.beta = beta

        self.M = np.random.randn(m, d) / np.sqrt(m)
        self.A = self.alpha * np.eye(self.m)
        self.b = np.zeros(self.m)
        self.theta_z = np.zeros(self.m)

    def _project(self, x):
        return self.M @ x

    def update(self, context, r):
        z = self._project(context)
        self.A += np.outer(z, z)
        self.b += r * z
        self.theta_z = np.linalg.solve(self.A, self.b)

    def score(self, context):
        z = self._project(context)
        mean = np.dot(z, self.theta_z)
        A_inv = np.linalg.inv(self.A)
        exp = np.sqrt(z @ A_inv @ z)
        return mean + self.beta * exp

    def select_arm(self, contexts):
        scores = [self.score(ctx) for ctx in contexts]
        return int(np.argmax(scores))




class LinUCBwithPSI_rank1:
    def __init__(self, d=10, epsilon=1.0, alpha=1.0, rank=10):
        self.d = d
        self.epsilon = epsilon
        self.sqrt_epsilon = 1 / sqrt(epsilon)
        self.alpha = alpha
        self.rank = rank

        self.U = np.zeros((d, 2 * rank))
        self.V = np.zeros((d, 2 * rank))
        self.n_cols = 0

        self.Ut = None
        self.St = None
        self.Vt = None

        self.b = np.zeros(self.d, dtype=np.float32)
        self.theta = np.zeros(self.d, dtype=np.float32)

    def _L_matvec(self, vec):
        L_0_inv_vec = self.sqrt_epsilon * vec.copy()
        if self.n_cols == 0:
            return L_0_inv_vec
        U = self.U[:, :self.n_cols]
        V = self.V[:, :self.n_cols]
        return L_0_inv_vec - U @ (V.T @ L_0_inv_vec)

    def update(self, context, reward):
        self.b += reward * context

        bar_x = self._L_matvec(vec=context)

        norm_bar_x_sq = norm(bar_x) ** 2
        if norm_bar_x_sq < 1e-12:
            self._update_theta()
            return

        alpha_t = (sqrt(1 + norm_bar_x_sq) - 1) / norm_bar_x_sq
        beta_t = alpha_t / (1 + alpha_t * norm_bar_x_sq)

        self._update_factors(bar_x=bar_x, beta_t=beta_t)
        self._update_theta()

    def _update_factors(self, bar_x, beta_t):
        delta_u = beta_t * bar_x

        if self.n_cols > 0:
            U = self.U[:, :self.n_cols]
            V = self.V[:, :self.n_cols]
            delta_v = bar_x - V @ (U.T @ bar_x)
        else:
            delta_v = bar_x

        self.U[:, self.n_cols] = delta_u
        self.V[:, self.n_cols] = delta_v
        self.n_cols += 1

        if self.n_cols == 2 * self.rank:
            if self.Ut is None:
                self.Ut, self.St, self.Vt = svd_U_V_T(
                    self.U[:, :self.rank], self.V[:, :self.rank], self.rank
                )

            delta_U_new = self.U[:, self.rank:self.n_cols]
            delta_V_new = self.V[:, self.rank:self.n_cols]
            self.Ut, self.St, self.Vt = integrator(
                self.Ut, self.St, self.Vt, delta_U_new, delta_V_new
            )


            US = self.Ut @ self.St
            self.U[:, :self.rank] = US
            self.V[:, :self.rank] = self.Vt
            self.n_cols = self.rank  #сбрасываем

    def _update_theta(self):
        b_eps = self.epsilon * self.b
        if self.n_cols == 0:
            self.theta = b_eps
            return

        U = self.U[:, :self.n_cols]
        V = self.V[:, :self.n_cols]

        Ub = U.T @ b_eps
        Vb = V.T @ b_eps
        self.theta = b_eps - V @ Ub - U @ Vb + V @ (U.T @ (U @ Vb))

    def score1(self, context):
        ctx = context
        mean = float(np.dot(self.theta.T, ctx))

        if self.n_cols == 0:
            diff_sq = np.dot(ctx, ctx)
        else:
            U = self.U[:, :self.n_cols]
            V = self.V[:, :self.n_cols]
            v = V.T @ ctx
            Uv = U @ v
            diff_sq = np.dot(ctx, ctx) - 2 * np.dot(ctx, Uv) + np.dot(Uv, Uv)

        exp = self.sqrt_epsilon * np.sqrt(max(diff_sq, 0))
        return mean + self.alpha * exp

    def score(self, context):
        ctx = context
        mean = float(np.dot(self.theta.T, ctx))
        U = self.U[:, :self.n_cols]
        V = self.V[:, :self.n_cols]
        v = V.T @ ctx
        exp = self.sqrt_epsilon * np.linalg.norm(ctx - (U @ v))
        return mean + self.alpha * exp

    def select_arm(self, contexts):
        scores = [self.score(ctx) for ctx in contexts]
        return int(np.argmax(scores))

def getStartingValues(u, v, k):
    Qu, Ru = np.linalg.qr(u)
    Qv, Rv = np.linalg.qr(v)


    small_matrix = Ru @ Rv.T

    try:
        U_s, S, Vh_s = np.linalg.svd(small_matrix, full_matrices=False)
    except np.linalg.LinAlgError:

        print('reg')
        reg = 1e-10 * np.eye(small_matrix.shape[0])
        small_matrix_reg = small_matrix + reg
        U_s, S, Vh_s = np.linalg.svd(small_matrix_reg, full_matrices=False)

    U_s = U_s[:, :k]
    S = S[:k]
    S = np.diag(S)
    Vh_s = Vh_s[:k, :]

    U = Qu @ U_s
    V = Qv @ Vh_s.T
    #print(U.shape)
    #print( U, S, V)

    return U, S, V




def symmetric_factorization_ambikassaran_qr(X_bar):
    d, B = X_bar.shape  # x_bar

    # print(X_Bar.shape)
    Q, R = np.linalg.qr(X_bar)
    # print(R.shape)

    T = np.eye(B) + R @ R.T
    M = np.linalg.cholesky(T)

    Y_tB = (M - np.eye(B))
    return Y_tB, Q

class LinUCBwithPSI_Batch:
    def __init__(self, d, epsilon=1.0, alpha=1.0, rank=10):
        self.d = d
        self.epsilon = epsilon
        self.eps = 1 / np.sqrt(epsilon)
        self.alpha = alpha
        self.rank = rank

        self.U = np.empty((d, 0), dtype=np.float32)
        self.V = np.empty((d, 0), dtype=np.float32)
        self.U_psi = None
        self.S_psi = None
        self.V_psi = None

        self.b = np.zeros(d, dtype=np.float32)
        self.theta = np.zeros(d, dtype=np.float32)

    def update(self, X_batch, rewards):

        self.b += X_batch @ rewards


        if self.U.shape[1] > 0:
            X_bar = (X_batch - self.U @ (self.V.T @ X_batch)) * self.eps
        else:
            X_bar = X_batch * self.eps

        try:
            Y_tB, Q = symmetric_factorization_ambikassaran_qr(X_bar)
            Y_tB_inv = np.linalg.inv(Y_tB + 1e-10 * np.eye(X_bar.shape[1]))
            C = np.linalg.inv(Y_tB_inv + np.eye(Y_tB.shape[0]))
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error: {e}")
            return False

        self._apply_psi_batch(Q, C)
        self._update_theta()
        return True

    def _apply_psi_batch(self, Q, C):
        # U_update = Q @ C, V_update = Q - V @ U^T @ Q
        U_update = Q @ C
        if self.V.shape[1] > 0:
            V_update = Q - self.V @ (self.U.T @ Q)
        else:
            V_update = Q

        current_cols = self.U.shape[1]

        if current_cols < self.rank:
            if current_cols > 0:
                self.U = np.column_stack([self.U, U_update])
                self.V = np.column_stack([self.V, V_update])
            else:
                self.U = U_update
                self.V = V_update
        else:
            if self.U_psi is None:
                self.U_psi, self.S_psi, self.V_psi = svd_U_V_T(self.U, self.V, self.rank)

            self.U_psi, self.S_psi, self.V_psi = integrator(
                self.U_psi, self.S_psi, self.V_psi, U_update, V_update
            )
            self.U = self.U_psi @ self.S_psi
            self.V = self.V_psi

    def _update_theta(self):
        b_eps = self.eps ** 2 * self.b
        if self.U.shape[1] == 0:
            self.theta = b_eps
            return

        Ub = self.U.T @ b_eps
        Vb = self.V.T @ b_eps
        self.theta = b_eps - self.V @ Ub - self.U @ Vb + self.V @ (self.U.T @ (self.U @ Vb))

    def score(self, context):
        mean = np.dot(self.theta, context)
        if self.U.shape[1] == 0:
            exp = self.eps * np.linalg.norm(context)
        else:
            v = self.V.T @ context
            exp = self.eps * np.linalg.norm(context - self.U @ v)
        return mean + self.alpha * exp

    def select_arm(self, contexts):
        scores = [self.score(ctx) for ctx in contexts]
        return int(np.argmax(scores))