from functools import wraps


def ema(data, gamma=0.9):
    # EMA(t) = gamma * x_t + (1 - gamma) * EMA(t-1)
    ema_ = 0
    for x in data:
        ema_ = gamma * x + (1 - gamma) * ema_
    return ema_