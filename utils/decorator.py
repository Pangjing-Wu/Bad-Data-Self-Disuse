import numpy as np
from functools import wraps

from.math import ema


# >>>>>>>>>> mean.
def mean_return(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return np.mean(ret)
    return wrapper

def mean_returns(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return tuple([np.mean(val) for val in ret])
    return wrapper
# <<<<<<<<<< mean.


# >>>>>>>>>> exponential moving average.
def ema_return(gamma):
    def ema_decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            return ema(ret, gamma)
        return wrapper
    return ema_decorate


def ema_returns(gamma):
    def ema_decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            return tuple([ema(x, gamma) for x in ret])
        return wrapper
    return ema_decorate
# <<<<<<<<<< exponential moving average.


# >>>>>>>>>> repeat.
def repeat(n):
    """return: [arg#1:1, ..., arg#1:N]"""
    def repeat_decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return [func(*args, **kwargs) for _ in range(n)]
        return wrapper
    return repeat_decorate


def repeats(n):
    """return: [[arg#1:1, ..., arg#1:N], [arg#2:1, ..., arg#2:N], ...]"""
    def repeat_decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return zip(*[func(*args, **kwargs) for _ in range(n)])
        return wrapper
    return repeat_decorate
# <<<<<<<<<< repeat.