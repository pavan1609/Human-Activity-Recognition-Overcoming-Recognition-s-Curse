import numpy as np
from numba import jit

@jit(nopython=True)
def dynamic_time_warping(x, y):
    n, m = x.shape[0], y.shape[0]
    cost = np.zeros((n, m))

    cost[0, 0] = np.linalg.norm(x[0] - y[0])

    for i in range(1, n):
        cost[i, 0] = cost[i-1, 0] + np.linalg.norm(x[i] - y[0])

    for j in range(1, m):
        cost[0, j] = cost[0, j-1] + np.linalg.norm(x[0] - y[j])

    for i in range(1, n):
        for j in range(1, m):
            min_cost = min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            cost[i, j] = np.linalg.norm(x[i] - y[j]) + min_cost

    return cost[-1, -1]  # Return only the distance
