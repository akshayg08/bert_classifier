from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                if "first_moment" not in state:
                    state["first_moment"] = torch.zeros(grad.shape).to(grad.device)
                if "second_moment" not in state:
                    state["second_moment"] = torch.zeros(grad.shape).to(grad.device)
                if "time_step" not in state:
                    state["time_step"] = 0

                beta_1 = group["betas"][0]
                beta_2 = group["betas"][1]

                state["first_moment"] = beta_1*state["first_moment"] + (1 - beta_1)*grad
                state["second_moment"] = beta_2*state["second_moment"] + (1 - beta_2)*(grad*grad) 
                state["time_step"] += 1
                self.state[p] = state

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                t = state["time_step"]
                alpha_t = alpha * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

                # Update parameters
                # Please note: you should update p.data (not p), to avoid an error about a leaf Variable being used in an in-place operation
                m_t = state["first_moment"]
                v_t = state["second_moment"]
                new_p_data = p.data - alpha_t*m_t/(torch.sqrt(v_t) + group["eps"])

                # Add weight decay after the main gradient-based updates.
                # Please note that, *unlike in https://arxiv.org/abs/1711.05101*, the learning rate should be incorporated into this update.
                # Please also note: you should update p.data (not p), to avoid an error about a leaf Variable being used in an in-place operation
                p.data = new_p_data - alpha*group["weight_decay"]*p.data

        return loss
