import torch


class LeastSquaresProxPointOptimizer:
    def __init__(self, x, step_size):
        self._x = x
        self._step_size = step_size

    def step(self, A_batch, b_batch):
        # helper variables
        x = self._x
        step_size = self._step_size
        m = A_batch.shape[0]  # number of rows = batch size

        # compute linear system coefficients
        P_batch = torch.addmm(torch.eye(m, dtype=A_batch.dtype), A_batch, A_batch.t(), beta=m, alpha=step_size)
        rhs = torch.addmv(b_batch, A_batch, x)

        # solve positive-definite linear system using Cholesky factorization
        P_factor = torch.cholesky(P_batch)
        rhs_chol = rhs.unsqueeze(1)
        s_star = torch.cholesky_solve(rhs_chol, P_factor)

        # perform step
        step_dir = torch.mm(A_batch.t(), s_star)
        x.sub_(step_size * step_dir.reshape(x.shape))

        # return the losses w.r.t the params before making the step
        return 0.5 * (rhs ** 2)
