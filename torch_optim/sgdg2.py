import math

import torch
from torch.optim.optimizer import Optimizer
from torch import flatten

from .utils import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("SGD_G2",)


class SGD_G2(Optimizer):
    """Implements SGD-G2 algorithm.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        delta: threhold that determines whether a set of parameters is scale
            invariant or not (default: 0.1)
        nesterov: enables Nesterov momentum (default: False)

    Note:
        Reference code: https://github.com/clovaai/AdamP
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
       
        
        defaults = dict(
            lr=lr,
            betas=betas,
        )
        super(SGD_G2, self).__init__(params, defaults)


    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                pbis = p
                pbis.data.add_(grad, alpha= - group["lr"]) #p-h*grad
                gradbis = pbis.grad.data
                
                beta = group["betas"][0]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                     

                state["step"] += 1
        
                prod = float(torch.dot((flatten(grad) - flatten(gradbis)),flatten(grad)))
                norm_square = float(torch.dot(flatten(grad - gradbis),flatten(grad-gradbis)))
                if prod >0:
                    step_size_opt = 2* group["lr"]*prod/norm_square 
                else:
                    step_size_opt = group["lr"]
                    
                if group["lr"] <= step_size_opt:
                    group["lr"] = beta*group["lr"] + (1-beta)*step_size_opt
                else:
                    group["lr"] =  (1-beta)*step_size_opt
                

                # Step
                p.data.add_(grad, alpha=- group["lr"] )

        return loss