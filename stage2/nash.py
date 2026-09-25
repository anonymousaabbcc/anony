import warnings
import cvxpy as cp
import numpy as np
import torch
import torch.distributed as dist
from .routing import project_nash_alpha

class DistributedNashMTL:

    def __init__(self, n_tasks: int, update_weights_every: int=1, normalize_mean: bool=True, optim_niter: int=20, solver_max_iters: int=100, eps: float=1e-10, alpha_floor: float=0.0):
        self.n_tasks = int(n_tasks)
        self.update_weights_every = max(1, int(update_weights_every))
        self.normalize_mean = bool(normalize_mean)
        self.optim_niter = int(optim_niter)
        self.solver_max_iters = int(solver_max_iters)
        self.eps = float(eps)
        self.alpha_floor = float(alpha_floor)
        if self.alpha_floor < 0:
            raise ValueError('alpha_floor must be >= 0')
        if self.normalize_mean and self.alpha_floor >= 1.0:
            raise ValueError('alpha_floor must be < 1 when normalize_mean=True')
        self.step = 0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)
        self.normalization_factor = np.ones((1,), dtype=np.float64)
        self._init_problem()

    def _init_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.n_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.n_tasks, self.n_tasks), value=np.eye(self.n_tasks))
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + 1 / G_prvs_alpha @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        G_alpha = self.G_param @ self.alpha_param
        constraints = [-cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i]) <= 0 for i in range(self.n_tasks)]
        objective = cp.Minimize(cp.sum(G_alpha) + phi_alpha / self.normalization_factor_param)
        self.problem = cp.Problem(objective, constraints)

    def _stop(self, gtg, alpha_t):
        return self.alpha_param.value is None or np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + self.eps)) < 0.001 or np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-06

    def _solve(self, gtg):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor
        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning, module='cvxpy')
                    self.problem.solve(solver=cp.ECOS, warm_start=True, max_iters=self.solver_max_iters)
            except Exception:
                self.alpha_param.value = self.prvs_alpha_param.value
            if self._stop(gtg, alpha_t):
                break
            alpha_t = self.alpha_param.value
        if alpha_t is not None and np.all(np.isfinite(alpha_t)) and np.all(alpha_t > 0):
            self.prvs_alpha = np.asarray(alpha_t, dtype=np.float32)
        return self.prvs_alpha

    @staticmethod
    def _distributed():
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _rank():
        return dist.get_rank() if DistributedNashMTL._distributed() else 0

    @staticmethod
    def _world_size():
        return dist.get_world_size() if DistributedNashMTL._distributed() else 1

    def local_task_gradients(self, losses: torch.Tensor, shared_parameters, *, preserve_graph: bool=False):
        if len(losses) != self.n_tasks:
            raise ValueError(f'Expected {self.n_tasks} losses, got {len(losses)}')
        params = list(shared_parameters)
        if not params:
            raise RuntimeError('Nash bargaining parameter list is empty')
        task_grads = []
        for task_idx, loss in enumerate(losses):
            grads = torch.autograd.grad(loss, params, retain_graph=preserve_graph or task_idx < self.n_tasks - 1, create_graph=False, allow_unused=False)
            task_grads.append([g.detach().float() for g in grads])
        return task_grads

    @staticmethod
    def accumulate_task_gradients_(accumulator, task_grads):
        if accumulator is None:
            return [[g.clone() for g in task] for task in task_grads]
        if len(accumulator) != len(task_grads):
            raise RuntimeError('Task-gradient accumulator task count mismatch')
        for acc_task, new_task in zip(accumulator, task_grads):
            if len(acc_task) != len(new_task):
                raise RuntimeError('Task-gradient accumulator parameter count mismatch')
            for a, g in zip(acc_task, new_task):
                a.add_(g)
        return accumulator

    def global_mean_accumulated_task_gradients(self, accumulated, local_micro_steps: int):
        if accumulated is None or local_micro_steps <= 0:
            raise RuntimeError('No accumulated Nash task gradients')
        denom = float(self._world_size() * int(local_micro_steps))
        out = []
        for task in accumulated:
            global_task = []
            for g in task:
                gg = g
                if self._distributed():
                    dist.all_reduce(gg, op=dist.ReduceOp.SUM)
                gg.div_(denom)
                global_task.append(gg)
            out.append(global_task)
        return out

    @staticmethod
    def _gram(task_grads):
        n = len(task_grads)
        device = task_grads[0][0].device
        gtg = torch.zeros((n, n), device=device, dtype=torch.float64)
        for i in range(n):
            for j in range(i, n):
                dot = torch.zeros((), device=device, dtype=torch.float64)
                for gi, gj in zip(task_grads[i], task_grads[j]):
                    dot = dot + torch.sum(gi.double() * gj.double())
                gtg[i, j] = dot
                gtg[j, i] = dot
        return gtg

    def should_update_now(self):
        return self.step % self.update_weights_every == 0

    def _rescale_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        return project_nash_alpha(alpha, normalize_mean=self.normalize_mean, alpha_floor=self.alpha_floor, eps=self.eps)

    def cached_weights_and_advance(self, device, dtype=torch.float32):
        alpha = torch.tensor(self.prvs_alpha, device=device, dtype=dtype)
        alpha = self._rescale_alpha(alpha)
        self.step += 1
        return alpha.detach()

    def weights_from_task_gradients(self, global_task_grads, device, dtype=torch.float32):
        should_update = self.should_update_now()
        if should_update:
            gtg = self._gram(global_task_grads)
            norm = torch.linalg.vector_norm(gtg)
            if not torch.isfinite(norm) or norm.item() <= self.eps:
                alpha_np = self.prvs_alpha
            else:
                self.normalization_factor = np.array([norm.item()], dtype=np.float64)
                normalized = (gtg / norm).detach().cpu().numpy()
                if self._rank() == 0:
                    alpha_np = self._solve(normalized)
                else:
                    alpha_np = np.empty(self.n_tasks, dtype=np.float32)
                if self._distributed():
                    alpha_t = torch.tensor(alpha_np, device=device, dtype=torch.float32)
                    dist.broadcast(alpha_t, src=0)
                    alpha_np = alpha_t.cpu().numpy()
                    self.prvs_alpha = alpha_np.copy()
        else:
            alpha_np = self.prvs_alpha
        alpha = torch.tensor(alpha_np, device=device, dtype=dtype)
        alpha = self._rescale_alpha(alpha)
        self.step += 1
        return alpha.detach()

    def weights(self, losses: torch.Tensor, shared_parameters):
        raise RuntimeError('Deprecated in split-Nash v3: do not form one alpha-weighted scalar loss. Use exact_nash_accumulated_update(), which routes Nash weights only to shared parameters and keeps city-private gradients unweighted.')

    def state_dict(self):
        return {'step': self.step, 'prvs_alpha': self.prvs_alpha.copy(), 'normalization_factor': self.normalization_factor.copy(), 'normalize_mean': self.normalize_mean, 'update_weights_every': self.update_weights_every, 'alpha_floor': self.alpha_floor}

    def load_state_dict(self, state):
        self.step = int(state.get('step', 0))
        self.prvs_alpha = np.asarray(state.get('prvs_alpha', np.ones(self.n_tasks)), dtype=np.float32)
        self.normalization_factor = np.asarray(state.get('normalization_factor', [1.0]), dtype=np.float64)
        self.normalize_mean = bool(state.get('normalize_mean', self.normalize_mean))
        self.update_weights_every = max(1, int(state.get('update_weights_every', self.update_weights_every)))
        self.alpha_floor = float(state.get('alpha_floor', self.alpha_floor))
        self.prvs_alpha_param.value = self.prvs_alpha

class EqualWeightSharedBargaining:

    def __init__(self, n_tasks: int):
        self.n_tasks = int(n_tasks)
        self.step = 0
        self.normalize_mean = True
        self.alpha_floor = 0.0

    def should_update_now(self):
        return False

    def cached_weights_and_advance(self, device, dtype=torch.float32):
        self.step += 1
        return torch.ones(self.n_tasks, device=device, dtype=dtype)

    def state_dict(self):
        return {'mode': 'equal', 'n_tasks': self.n_tasks, 'step': self.step}

    def load_state_dict(self, state):
        self.step = int(state.get('step', 0))
