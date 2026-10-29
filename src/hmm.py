"""Gaussian-emission Hidden Markov Model implemented from scratch with numpy.

Parameters
----------
pi  : (K,)    initial state distribution
A   : (K, K)  transition matrix, A[i, j] = P(z_t = j | z_{t-1} = i)
mu  : (K, D)  emission means
var : (K, D)  emission variances (diagonal covariance, floored for stability)

All recursions are done in log-space (log-sum-exp) so long sequences cannot underflow.
Supervised parts : initialisation of pi, A, mu, var from labelled training sequences.
Unsupervised part: Baum-Welch (EM) refinement, which ignores the labels.
"""
import numpy as np
from scipy.special import logsumexp

LOG0 = -1e300


def _log(x):
    """log that maps 0 to a very negative finite number."""
    with np.errstate(divide="ignore"):
        return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), LOG0)


class GaussianHMM:
    def __init__(self, n_states, var_floor=1e-2, trans_floor=1e-3):
        self.K = n_states
        self.var_floor = var_floor        # minimum per-feature variance (Z-score units)
        self.trans_floor = trans_floor    # minimum transition probability (smoothing)

    # ------------------------------------------------------------------ emissions
    def log_emission(self, X):
        """(T, K) matrix of log N(x_t | mu_k, diag(var_k))."""
        D = X.shape[1]
        d2 = ((X[:, None, :] - self.mu[None]) ** 2 / self.var[None]).sum(-1)
        return -0.5 * (D * np.log(2 * np.pi) + np.log(self.var).sum(1)[None] + d2)

    # ------------------------------------------------------------------ supervised init
    def init_supervised(self, seqs, labels):
        """Initialise pi, A, mu, var from labelled sequences (lists of (T,D) arrays / (T,) int arrays)."""
        K = self.K
        X = np.vstack(seqs); y = np.concatenate(labels)
        self.mu = np.vstack([X[y == k].mean(0) for k in range(K)])
        self.var = np.maximum(np.vstack([X[y == k].var(0) for k in range(K)]), self.var_floor)
        pi = np.zeros(K); A = np.zeros((K, K))
        for l in labels:
            pi[l[0]] += 1
            np.add.at(A, (l[:-1], l[1:]), 1)
        self.pi = (pi + self.trans_floor) / (pi + self.trans_floor).sum()
        A = A + self.trans_floor
        self.A = A / A.sum(1, keepdims=True)
        return self

    # ------------------------------------------------------------------ forward / backward
    def forward(self, logB):
        """log-alpha (T,K) and log-likelihood of the sequence."""
        T, K = logB.shape
        la = np.empty((T, K)); lA = _log(self.A)
        la[0] = _log(self.pi) + logB[0]
        for t in range(1, T):
            la[t] = logsumexp(la[t - 1][:, None] + lA, axis=0) + logB[t]
        return la, logsumexp(la[-1])

    def backward(self, logB):
        """log-beta (T,K)."""
        T, K = logB.shape
        lb = np.zeros((T, K)); lA = _log(self.A)
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(lA + (logB[t + 1] + lb[t + 1])[None], axis=1)
        return lb

    def loglik(self, X):
        return self.forward(self.log_emission(X))[1]

    # ------------------------------------------------------------------ Viterbi
    def viterbi(self, X):
        """Most likely state sequence (log-space). Returns (path, log-prob of path)."""
        logB = self.log_emission(X); T, K = logB.shape; lA = _log(self.A)
        delta = np.empty((T, K)); psi = np.zeros((T, K), int)
        delta[0] = _log(self.pi) + logB[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + lA
            psi[t] = cand.argmax(0); delta[t] = cand.max(0) + logB[t]
        path = np.empty(T, int); path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, delta[-1].max()

    # ------------------------------------------------------------------ Baum-Welch (EM)
    def baum_welch(self, seqs, max_iter=200, tol=1e-4, verbose=False):
        """EM over several sequences. Stops when the total log-likelihood improves by < tol
        (or after max_iter). Returns the log-likelihood history."""
        hist = []
        for it in range(max_iter):
            K, D = self.K, seqs[0].shape[1]
            pi_n = np.zeros(K); A_n = np.zeros((K, K))
            g_sum = np.zeros(K); mu_n = np.zeros((K, D)); sq_n = np.zeros((K, D)); ll = 0.0
            lA = _log(self.A)
            for X in seqs:                                   # ---- E-step
                logB = self.log_emission(X)
                la, l = self.forward(logB); lb = self.backward(logB); ll += l
                lg = la + lb - l; g = np.exp(lg)             # gamma_t(k)
                pi_n += g[0]
                if len(X) > 1:                               # xi_t(i,j) summed over t
                    lx = la[:-1, :, None] + lA[None] + (logB[1:] + lb[1:])[:, None, :] - l
                    A_n += np.exp(lx).sum(0)
                g_sum += g.sum(0); mu_n += g.T @ X; sq_n += g.T @ (X ** 2)
            hist.append(ll)
            if verbose: print(f"iter {it:3d}  loglik {ll:.4f}")
            if it > 0 and abs(hist[-1] - hist[-2]) < tol:    # convergence check
                break
            # ---- M-step
            self.pi = (pi_n + 1e-12) / (pi_n + 1e-12).sum()
            A_n = A_n + self.trans_floor
            self.A = A_n / A_n.sum(1, keepdims=True)
            self.mu = mu_n / g_sum[:, None]
            self.var = np.maximum(sq_n / g_sum[:, None] - self.mu ** 2, self.var_floor)
        self.n_iter_ = len(hist)
        return np.array(hist)


# ====================================================================== sanity tests
def _brute_force(model, X):
    """Exhaustive log-likelihood and best path for tiny problems (K^T paths)."""
    import itertools
    logB = model.log_emission(X); T, K = logB.shape
    lA, lpi = _log(model.A), _log(model.pi)
    best, bp, scores = -np.inf, None, []
    for path in itertools.product(range(K), repeat=T):
        s = lpi[path[0]] + logB[0, path[0]]
        for t in range(1, T): s += lA[path[t - 1], path[t]] + logB[t, path[t]]
        scores.append(s)
        if s > best: best, bp = s, np.array(path)
    return logsumexp(scores), bp, best


def run_sanity_tests(seed=0):
    """Small automated checks used by the notebook. Raises AssertionError on failure."""
    rng = np.random.default_rng(seed)
    K, D = 3, 2
    m = GaussianHMM(K); m.mu = rng.normal(size=(K, D)) * 2; m.var = np.ones((K, D))
    m.pi = np.array([.5, .3, .2]); m.A = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7]])
    X = rng.normal(size=(6, D)) * 2
    ll_bf, path_bf, _ = _brute_force(m, X)
    assert np.isclose(m.loglik(X), ll_bf), "forward != brute force"
    la, _ = m.forward(m.log_emission(X)); lb = m.backward(m.log_emission(X))
    assert np.allclose(logsumexp(la + lb, axis=1), ll_bf), "alpha*beta != likelihood at every t"
    assert (m.viterbi(X)[0] == path_bf).all(), "Viterbi != brute force"
    # EM on synthetic data: A rows sum to 1, log-likelihood non-decreasing
    seqs = [rng.normal(size=(40, D)) + np.repeat(rng.normal(size=(1, D)) * 3, 40, 0) for _ in range(4)]
    e = GaussianHMM(K); lab = [np.repeat(np.arange(K), 14)[:40] for _ in seqs]
    e.init_supervised(seqs, lab); h = e.baum_welch(seqs, max_iter=30, tol=1e-9)
    assert np.allclose(e.A.sum(1), 1) and np.isclose(e.pi.sum(), 1), "A / pi not stochastic"
    assert (np.diff(h) >= -1e-6).all(), "EM log-likelihood decreased"
    return "all sanity tests passed"

if __name__ == "__main__":
    print(run_sanity_tests())
