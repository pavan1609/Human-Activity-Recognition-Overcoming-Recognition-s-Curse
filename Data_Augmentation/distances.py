import numpy as np

def dynamic_time_warping(x, y):
    n = x.shape[0]
    m = y.shape[0]
    cost = np.zeros((n, m))

    cost[0, 0] = np.abs(x[0] - y[0])

    for i in range(1, n):
        cost[i, 0] = cost[i-1, 0] + np.abs(x[i] - y[0])

    for j in range(1, m):
        cost[0, j] = cost[0, j-1] + np.abs(x[0] - y[j])

    for i in range(1, n):
        for j in range(1, m):
            minimum_cost = min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            cost[i, j] = np.abs(x[i] - y[j]) + minimum_cost

    return cost[-1, -1]
