import numpy as np
from numpy.linalg import svd


class CBSCFD:
    def __init__(self, num_arms, lambd, beta, m, d):
        self.num_arms = num_arms
        self.d = d
        self.beta = beta
        self.alpha = lambd
        self.m = m

        self.theta = {a: np.zeros(self.d) for a in range(num_arms)}
        self.Z = {a: np.zeros((0, self.d)) for a in range(num_arms)}
        self.H = {a: None for a in range(num_arms)}
        self.H_prev = {a: None for a in range(num_arms)}
        self.sum_xy = {a: np.zeros(self.d) for a in range(num_arms)}
        self.alpha_per_arm = {a: lambd for a in range(num_arms)}
        self.t = 0

    def _apply_V_inv(self, x, a):
        if self.Z[a].shape[0] == 0:
            return x / self.alpha_per_arm[a]
        Zx = self.Z[a] @ x
        HZx = self.H[a] @ Zx
        ZT_HZx = self.Z[a].T @ HZx
        return (x - ZT_HZx) / self.alpha_per_arm[a]

    def update(self, user_context, a, r):
        self.t += 1
        x_t = user_context
        self.sum_xy[a] += x_t * r
        n_prev = self.Z[a].shape[0]
        self.Z[a] = np.vstack([self.Z[a], x_t.reshape(1, -1)])
        n = self.Z[a].shape[0]

        if n == 2 * self.m:
            self._svd_update(a)
        else:
            self._sequential_update(x_t, n_prev, a)

        self.theta[a] = self._apply_V_inv(self.sum_xy[a], a)

    def _svd_update(self, a):
        U, Sigma, Vt = svd(self.Z[a], full_matrices=False)
        if len(Sigma) >= self.m:
            delta_t = Sigma[self.m - 1] ** 2
        else:
            delta_t = 0
        self.alpha_per_arm[a] += delta_t
        Sigma_hat_sq = np.maximum(Sigma[:self.m] ** 2 - delta_t, 0)
        Sigma_hat = np.sqrt(Sigma_hat_sq)
        self.Z[a] = Sigma_hat.reshape(-1, 1) * Vt[:self.m, :]
        H_diag = 1.0 / (Sigma_hat_sq + self.alpha_per_arm[a])
        self.H[a] = np.diag(H_diag)
        self.H_prev[a] = self.H[a].copy()

    def _sequential_update(self, x_t, n_prev, a):
        n = self.Z[a].shape[0]
        if n == 1:
            k = np.dot(x_t, x_t) + self.alpha_per_arm[a]
            self.H[a] = np.array([[1.0 / k]])
            self.H_prev[a] = self.H[a]
        else:
            if self.H_prev[a] is None or self.H_prev[a].shape[0] != n_prev:
                self.H_prev[a] = np.eye(n_prev) / self.alpha_per_arm[a]
            Z_prev = self.Z[a][:n_prev, :]
            Z_prev_x = Z_prev @ x_t
            p = self.H_prev[a] @ Z_prev_x
            k_scalar = np.dot(x_t, x_t) - np.dot(Z_prev_x, p) + self.alpha_per_arm[a]
            self.H[a] = np.zeros((n, n))
            self.H[a][:n-1, :n-1] = self.H_prev[a] + np.outer(p, p) / k_scalar
            self.H[a][:n-1, n-1] = -p / k_scalar
            self.H[a][n-1, :n-1] = -p / k_scalar
            self.H[a][n-1, n-1] = 1.0 / k_scalar
            self.H_prev[a] = self.H[a]

    def score(self, user_context, arm):
        x = user_context
        mean = np.dot(self.theta[arm], x)
        var = np.dot(x, self._apply_V_inv(x, arm))
        return mean + self.beta * np.sqrt(max(var, 0))

    def select_arm(self, contexts):
        scores = []
        for a in range(self.num_arms):
            x = contexts[a]
            mean = np.dot(self.theta[a], x)
            var = np.dot(x, self._apply_V_inv(x, a))
            scores.append(mean + self.beta * np.sqrt(max(var, 0)))
        return int(np.argmax(scores))


from collections import defaultdict
class LinUCB:
    def __init__(self, num_arms, d, alpha, epsilon=1.0):
        self.num_arms = num_arms
        self.d = d

        self.alpha = alpha
        self.epsilon = epsilon


        self.D = defaultdict(list)
        self.b = defaultdict(list)

        self.A = defaultdict(lambda: self.epsilon * np.eye(self.d))
        self.A_inv = defaultdict(lambda: np.eye(self.d) / self.epsilon)
        self.rhs = defaultdict(lambda: np.zeros(self.d))
        self.theta = defaultdict(lambda: np.zeros(self.d, dtype=np.float32))



    def append_interaction(self, user_context, a, r):
        x_ua = user_context
        self.D[a].append(x_ua)
        self.b[a].append(r)

    def batch_update(self, arms=None):
        if arms is None:

            arms = self.D.keys()

        for a in arms:
            if len(self.D[a]) == 0:
                continue

            D_a = np.vstack(self.D[a])
            b_a = np.array(self.b[a])

            self.A[a] += D_a.T @ D_a
            self.A_inv[a] = np.linalg.inv(self.A[a])
            self.rhs[a] += D_a.T @ b_a

            self.theta[a] = self.A_inv[a] @ self.rhs[a]


        self.D.clear()
        self.b.clear()

    def score(self, user_context, arm):
        ctx = user_context
        mean = float(np.dot(self.theta[arm].T, ctx))
        exp = self.alpha * np.sqrt(np.dot(ctx.T, self.A_inv[arm] @ ctx))
        return mean + exp

import numpy as np
from collections import defaultdict
from numpy.linalg import multi_dot
from scipy.linalg import svd, qr, norm
from math import sqrt

def integrator(tilde_Ut, S, tilde_Vt, delta_U, delta_V):
    #print("Integrator called")
    Ut = tilde_Ut.copy()
    Vt = tilde_Vt.copy()
    K1 = Ut @ S + delta_U.dot(delta_V.T.dot(Vt))
    tilde_U1, tilde_S1 = qr(K1,mode = "economic")
    tilde_S0 = tilde_S1 - tilde_U1.T.dot(delta_U.dot(delta_V.T.dot(Vt)))
    L1 = Vt.dot(tilde_S0.T) + delta_V.dot(delta_U.T.dot(tilde_U1))
    tilde_V1, S1 = qr(L1, mode = "economic")
    S1 = S1.T
    return tilde_U1, S1, tilde_V1

def svd_U_V_T(U, V, rank):
    Q_U, R_U = qr(U, mode = "economic")
    Q_V, R_V = qr(V, mode = "economic")

    U_svd, S_svd, V_svd = svd(R_U.dot(R_V.T), full_matrices = False)

    U_svd = U_svd[:, :rank]
    S_svd = S_svd[:rank]
    V_svd = V_svd[:rank, :]

    U_new = Q_U.dot(U_svd)
    V_new = V_svd.dot(Q_V.T)

    return U_new, np.diag(S_svd), V_new.T



class LinUCBwithPSI_rank1:
    def __init__(self, n_arms, d = 10, epsilon = 1.0, alpha = 1.0, rank = 10):
        self.n_arms = n_arms # количество ручек
        self.d = d # размерность контекстных векторов
        self.epsilon = epsilon # регуляризация
        self.sqrt_epsilon = 1 / sqrt(epsilon) # коэффициент для L_0^{-1}

        self.alpha = alpha # эскплорейшен
        self.rank = rank # малоранговое приближение

        self.U = defaultdict(lambda: np.empty((self.d, 0), dtype=np.float32))
        self.V = defaultdict(lambda: np.empty((self.d, 0), dtype=np.float32))

        self.first_time = defaultdict(lambda: False)
        #для  интегратора храним факторы с предыдущего шага
        self.Ut = defaultdict(lambda: None)
        self.St = defaultdict(lambda: None)
        self.Vt = defaultdict(lambda: None)

        self.b = defaultdict(lambda: np.zeros(self.d, dtype=np.float32)) # вектор наград
        self.theta = defaultdict(lambda: np.zeros(self.d, dtype=np.float32)) # оценка параметра

    def _L_matvec(self, vec, arm):

        L_0_inv_vec = self.sqrt_epsilon * vec.copy()
        return L_0_inv_vec - self.U[arm].dot(self.V[arm].T.dot(L_0_inv_vec))

    def update(self, user_context, arm, reward):

        self.b[arm] += reward * user_context

        bar_x = self._L_matvec(vec = user_context, arm = arm)

        norm_bar_x_sq = norm(bar_x) ** 2
        alpha_t = (sqrt(1 + norm_bar_x_sq) - 1) / norm_bar_x_sq
        beta_t = alpha_t / (1 + alpha_t * norm_bar_x_sq)

        self.U[arm], self.V[arm] = self._update_factors(bar_x=bar_x, beta_t=beta_t, arm=arm)

        self._update_theta(arm)

    def _update_u_and_v(self, x_bar, beta_t, arm):
        delta_u = beta_t * x_bar

        if self.V[arm].shape[1] > 0:
            delta_v = x_bar - self.V[arm] @ (self.U[arm].T @ x_bar)
        else:
            delta_v = x_bar

        return delta_u.reshape(-1, 1), delta_v.reshape(-1, 1)

    def _update_factors(self, bar_x, beta_t, arm):
        delta_U, delta_V = self._update_u_and_v(bar_x, beta_t, arm)
        if self.U[arm].shape[1] == 0:
            self.U[arm] = delta_U
            self.V[arm] = delta_V
            return self.U[arm], self.V[arm]
        elif self.U[arm].shape[1] < self.rank:
            self.U[arm] = np.column_stack([self.U[arm], delta_U])
            self.V[arm] = np.column_stack([self.V[arm], delta_V])
            return self.U[arm], self.V[arm]

        if self.Ut[arm] is None:
            self.Ut[arm], self.St[arm], self.Vt[arm] = svd_U_V_T(self.U[arm], self.V[arm], self.rank)
            #self.first_time[arm] = True
        self.Ut[arm], self.St[arm],  self.Vt[arm] = integrator(self.Ut[arm], self.St[arm], self.Vt[arm], delta_U, delta_V)

        return self.Ut[arm] @ self.St[arm],  self.Vt[arm]

    def _update_theta(self, arm):
        b_eps = self.epsilon * self.b[arm]

        term1 = b_eps
        term2 = self.V[arm].dot(self.U[arm].T.dot(b_eps))
        term3 = self.U[arm].dot(self.V[arm].T.dot(b_eps))
        term4 = self.V[arm].dot(self.U[arm].T.dot(term3))

        self.theta[arm] = term1 - term2 - term3 + term4

    def score(self, user_context, arm):
        """
        Compute the PSI-UCB score for a given user context and arm.
        """
        ctx = user_context
        mean = float(np.dot(self.theta[arm].T, ctx))
        v = self.V[arm].T @ ctx
        exp = (
            self.sqrt_epsilon
            * np.linalg.norm(ctx - (self.U[arm] @ v))
        )
        return mean + self.alpha * exp

