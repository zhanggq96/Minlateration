# https://stackoverflow.com/questions/17009774/quadratic-program-qp-solver-that-only-depends-on-numpy-scipy

import sys
import numpy as np
import scipy
from scipy import optimize as opt
# scipy has no module optimize ???????????????????
# https://stackoverflow.com/questions/37107627/python-problems-with-scipy-optimize-curve-fit
from plot_circles import plot_circles


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
            [np.abs(np.linalg.norm(p-x) - r) for x, r in zip(x_array_list, r_list)]
        )

        return l

    def jacobian(p):
        grad_j = np.zeros_like(p)
        for x0, r0 in zip(x_array_list, r_list):
            d_sum = np.linalg.norm(p-x0)
            grad_j += 2*(d_sum-r0) * (1/2)*d_sum**(-1) * 2*(p-x0)
        # row_multiple = 2*(d_sum - r) * 1/2*d_sum**(-1/2)
        return grad_j

    return loss_func, jacobian


def single_multilateration():
    circles = [
        [[3, 4], 1.2, None],
        [[3, 7], 3, None],
        [[5, 4], 1, None],
        [[0, 0], 5.66, None],

        [[8, 6], 1, None],
        [[7.5, 6], 1.5, None],
    ]

    # circles = [
    #     [[4, 4], 2, None],
    #     [[4.5, 4], 1.5, None],
    # ]
    #
    # circles = [
    #     [[28, 4], 1.5, None],
    #     [[26, 3], 2, None],
    #     [[25.77, 5.5], 2.755, None],
    # ]
    # print(zip(*circles))
    M = 20
    xlim = (0, M)
    ylim = (0, M)

    loss_func, grad_j = opt_func_dec(*zip(*circles))
    print(loss_func(np.array([0, 0]).T))

    options = {'disp': True}
    p0_example = np.array([2, 2]).T
    min_fun_vals = {
        'fun': sys.maxsize,
        'p': None
    }
    iters = 12

    for _ in range(iters):
        # p0 = np.array([7, 7]).T
        p0 = np.random.uniform(0, M, p0_example.shape)
        p = opt.minimize(loss_func, p0, jac=grad_j, method='SLSQP', options=options)
        # print(p)
        print(p.fun, p.x)
        if p.fun < min_fun_vals['fun']:
            min_fun_vals['fun'] = p.fun
            min_fun_vals['p'] = p.x

    plot_circles(circles, [min_fun_vals,], xlim=xlim, ylim=ylim, clear_dir_on_new=True)




if __name__ == '__main__':
    single_multilateration()