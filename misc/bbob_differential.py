import torch as t
import math

import bbobtorch
from bbobtorch import Problem
utils = bbobtorch.utils


@utils.seedable
def create_f24(dim, dev=None):
    """
    Creates a differentiable version of the BBOB function f24.

    Args:
        dim: Dimensionality of the problem.
        dev: Device to use.

    Returns:
        A differentiable version of the BBOB function f24.
    """
    s = 1 - 1/(2*math.sqrt(dim+20.0)-8.2)
    d = 1
    mu_0 = 2.5
    mu_1 = -math.sqrt((mu_0 ** 2 - d)/s)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(100, dim, t.float32, dev)
    ones = t.ones(size=(dim,), dtype=t.float32, device=dev)
    f_opt = utils.rand_fopt()
    x_opt = mu_0 / 2.0 * t.bernoulli(t.full((dim,), 0.5, dtype=t.float32, device=dev)) * 2.0 - 1.0
    def _f(x, R, Q, lamb, ones, f_opt, x_opt):
        x_hat = 2 * t.sign(x_opt)[None,:] * x
        z = (x_hat - mu_0) @ R @ lamb @ Q
        min1 = t.sum(t.pow(x_hat - mu_0, 2), dim=-1)
        min2 = d*dim + s * t.sum(t.pow(x_hat - mu_1, 2), dim=-1)
        m = t.where(min1<min2, min1, min2)#t.min(min1, min2, out=min1)
        second = 10 * (dim - t.sum(t.cos(2*utils.PI*z), dim=-1))#, out=min2))
        second += 10 ** 4 * utils.f_pen(x)
        second += f_opt
        second += m
        return second
    return Problem(
        _f, [R, Q, lamb, ones, f_opt, x_opt], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


def T_asy(x, beta, dim) -> t.Tensor:
    res = t.clone(x)
    pow = t.arange(0, dim, dtype=x.dtype, device=x.device)
    pow = 1 + beta * pow[None, :] / (dim - 1) * t.sqrt(x)
    pow = x**pow#t.pow(x, pow)#, out=pow)
    res[res > 0] = pow[res > 0]
    return res

@utils.seedable
def create_f03(dim, dev=None) -> Problem:
    """
    Creates a differentiable version of the BBOB function f03.

    Args:
        dim: Dimensionality of the problem.
        dev: Device to use.

    Returns:
        A differentiable version of the BBOB function f03.
    """
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    lamb = utils.Lambda(10, dim, t.float32, dev)
    def _f(x, x_opt, f_opt, lamb):
        z_tmp = T_asy(
            utils.T_osz(x - x_opt),
            0.2,
            dim,
        )
        z = (lamb @ z_tmp.T).T
        first_part = 10 * (dim - t.sum(t.cos(2*utils.PI*z), dim=-1))
        z_norm = t.sum(z * z, dim=-1)
        return first_part + z_norm + f_opt
    return Problem(
        _f, [x_opt, f_opt, lamb], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )

@utils.seedable
def create_f01(dim, dev=None) -> Problem:
    """
    Creates a differentiable version of the BBOB function f01.

    Args:
        dim: Dimensionality of the problem.
        dev: Device to use.

    Returns:
        A differentiable version of the BBOB function f01.
    """
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    def _f(x, x_opt, f_opt):
        z = x - x_opt[None, :]
        norm = t.sum(z * z, dim=-1)
        return norm + f_opt
    return Problem(
        _f, [x_opt, f_opt], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )