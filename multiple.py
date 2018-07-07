# https://stackoverflow.com/questions/17009774/quadratic-program-qp-solver-that-only-depends-on-numpy-scipy

import sys
import numpy as np
import scipy
from scipy import optimize as opt
# scipy has no module optimize ???????????????????
# https://stackoverflow.com/questions/37107627/python-problems-with-scipy-optimize-curve-fit
from plot_circles import plot_circles
from copy import deepcopy


def single_loss(p, x, r):
    return np.abs(np.linalg.norm(p-x) - r)

def opt_func_dec(x_list, r_list, lat_id_list):
    # x_list: a vector of x's
    # x represents [x y]' : a vector of positions
    # r_list: a vector of r's
    # r represents r: radius of circle corresponding to position

    x_list = list(x_list)
    r_list = list(r_list)

    x_array_list = []
    for x, y in x_list:
        x_array_list.append(np.array([x, y]).T)
    print(x_array_list)

    # https://stackoverflow.com/questions/4582521/python-creating-dynamic-functions
    def loss_func(p):
        # | ||p-x}| - r |^2
        l = np.sum(
            [single_loss(p, x, r) for x, r in zip(x_array_list, r_list)]
        )

        return l

    def jacobian(p):
        grad_j = np.zeros_like(p)
        for x0, r0 in zip(x_array_list, r_list):
            d_sum = np.linalg.norm(p-x0)
            grad_j += 2*(d_sum-r0) * (1/2)*d_sum**(-1/2) * 2*(p-x0)
        # row_multiple = 2*(d_sum - r) * 1/2*d_sum**(-1/2)
        return grad_j

    return loss_func, jacobian


def multiple_multilateration():
    circles_orig = [
        [[3, 4], 1.2, 0],
        [[3, 7], 3, 0],
        [[5, 4], 1, 0],
        [[0, 0], 5.66, 0],
        [[7.5, 6], 2, 1],
        [[8, 6], 1.5, 1],
    ]

    circles = deepcopy(circles_orig)
    iters = 12
    lat_clusters = 2
    # print(zip(*circles))

    def multilat(data):

        loss_func, grad_j = opt_func_dec(*zip(*data))
        print(loss_func(np.array([0, 0]).T))

        options = {'disp': True}
        p0_example = np.array([2, 2]).T
        min_fun_vals = {
            'fun': sys.maxsize,
            'p': None
        }

        for _ in range(iters):
            # p0 = np.array([7, 7]).T
            p0 = np.random.uniform(0, 10, p0_example.shape)
            p = opt.minimize(loss_func, p0, jac=grad_j, method='SLSQP', options=options)
            # print(p)
            # print(p.fun, p.x)
            if p.fun < min_fun_vals['fun']:
                min_fun_vals['fun'] = p.fun
                min_fun_vals['p'] = p.x

        print(min_fun_vals)
        return min_fun_vals

    def argmin(circle, min_fun_vals):
        min_val = sys.maxsize
        min_lat_cluster = None
        circle_center = circle[0]
        circle_radius = circle[1]
        for min_fun_val in min_fun_vals:
            # p, x, r
            loss = single_loss(min_fun_val['p'], np.array(circle_center).T, circle_radius)
            if min_val < loss:
                min_val = loss
                min_lat_cluster = min_fun_val['index']

        return min_lat_cluster

    def reassign(circles, min_fun_vals):

        for i, ((x, y), r, lat_cluster_id) in enumerate(circles):
            min_lat_cluster = argmin(circles[i], min_fun_vals)
            circles[i][2] = min_lat_cluster

        return +1

    lat_clusters = [[] for _ in range(lat_clusters)]
    for i, ((x, y), r, lat_cluster_id) in enumerate(circles):
        lat_clusters[lat_cluster_id].append(circles[i])

    min_fun_vals_list = []
    for i, lat_cluster in enumerate(lat_clusters):
        min_fun_vals = multilat(lat_cluster)
        min_fun_vals['index'] = i
        min_fun_vals_list.append(min_fun_vals)

    # circles_list: list of lateration clusters with circles in them

    print(circles)
    print(min_fun_vals_list)
    plot_circles(circles, min_fun_vals_list)
    reassign(circles, min_fun_vals_list)
    print(circles)

multiple_multilateration()