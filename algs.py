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