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

def single_loss_ord(p, x, r, norm_ord=2):
    return np.abs(np.linalg.norm(p-x, ord=norm_ord) - r) ** norm_ord

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

    def loss_func_ord(p, norm_ord=2):
        l = np.sum(
            [single_loss_ord(p, x, r, norm_ord=norm_ord) for x, r in zip(x_array_list, r_list)]
        )

        return l
    # https://stackoverflow.com/questions/4582521/python-creating-dynamic-functions
    def loss_func(p):
        # | ||p-x}| - r |^2
        l = sum(
            (single_loss(p, x, r) for x, r in zip(x_array_list, r_list))
        )

        return l

    def jacobian(p):
        grad_j = np.zeros_like(p)
        for xi, ri in zip(x_array_list, r_list):
            d = np.linalg.norm(p-xi)
            grad_j += 2*(d-ri) * (1/2)*(1/d) * 2*(p-xi)
        return grad_j

    return loss_func_ord, loss_func, jacobian


def multiple_multilateration(circles_orig, xlim=(0,10), ylim=(0,10),
                             num_lat_clusters=2, opt_iters=7, recluster_iters=5):
    # circles_orig = [
    #     [[3, 4], 1.2, 0],
    #     [[3, 7], 3, 0],
    #     [[5, 4], 1, 0],
    #     [[0, 0], 5.66, 0],
    #     [[7.5, 6], 2, 1],
    #     [[8, 6], 1.5, 1],
    # ]

    circles_copy = deepcopy(circles_orig)
    print('[multiple_multilateration] circles_copy init:', circles_copy)


    # print(zip(*circles_copy))
    options = {'disp': False}
    p0_example = np.array([2, 2]).T
    # print(loss_func(np.array([0, 0]).T))

    def get_local_lims(circles):
        max_x = -sys.maxsize
        min_x = +sys.maxsize
        max_y = -sys.maxsize
        min_y = +sys.maxsize
        for (x, y), r, lat_cluster_id in circles:
            max_x = max(x + r, max_x)
            min_x = min(x - r, min_x)
            max_y = max(y + r, max_y)
            min_y = min(y - r, min_x)

        return (min_x, max_x), (min_y, max_y)

    def multilat(circles, use_local_lims=False):

        loss_func_ord, loss_func, grad_j = opt_func_dec(*zip(*circles))

        min_fun_vals = {
            'loss': sys.maxsize,
            'p': None,
            'circles': deepcopy(circles)
        }

        # Get coordinate limits of the cluster's data
        if use_local_lims:
            cluster_xlim, cluster_ylim = get_local_lims(circles)
        else:
            cluster_xlim, cluster_ylim = xlim, ylim

        for _ in range(opt_iters):
            # p0 = np.array([7, 7]).T
            # p0 = np.random.uniform(0, 30, p0_example.shape)

            # Generate single random initial cluster center
            p0 = np.array([np.random.uniform(*cluster_xlim), np.random.uniform(*cluster_ylim)]).T
            # Optimize over this
            p = opt.minimize(loss_func, p0, jac=grad_j, method='SLSQP', options=options)
            # print(p)
            # print(p.fun, p.x)
            if p.fun < min_fun_vals['loss']:
                min_fun_vals['loss'] = p.fun
                min_fun_vals['p'] = p.x

        # print(min_fun_vals)
        return min_fun_vals

    def argmin_p(circle, min_fun_vals):
        min_val = sys.maxsize
        second_min_lat_cluster = sys.maxsize
        min_lat_cluster = None
        circle_center = circle[0]
        circle_radius = circle[1]
        for min_fun_val in min_fun_vals:
            # p, x, r
            loss = single_loss(min_fun_val['p'], np.array(circle_center).T, circle_radius)
            if loss < min_val:
                min_val = loss
                second_min_lat_cluster = min_lat_cluster
                min_lat_cluster = min_fun_val['index']

        return min_lat_cluster, second_min_lat_cluster

    def argmax_x(min_fun_val, circles):
        max_val = -sys.maxsize
        max_delocalized_circle = None
        max_delocalized_circle_index = None
        p = min_fun_val['p']

        for i, ((x, y), r, lat_cluster_id) in enumerate(circles):
            # confusing notation :(
            x = np.array([x, y]).T
            loss = single_loss(p, x, r)
            if loss > max_val:
                max_val = loss
                max_delocalized_circle = circles[i]
                max_delocalized_circle_index = i

        return max_delocalized_circle, max_delocalized_circle_index

    def reassign_circle_clusters(circles, min_fun_vals, epsilon=0.25):

        for i, ((x, y), r, lat_cluster_id) in enumerate(circles):
            # second_min_lat_cluster unused - might be useful later for prob. epsilon swap
            min_lat_cluster, _ = argmin_p(circles[i], min_fun_vals)
            circles[i][2] = min_lat_cluster

        return +1

    best_total_loss = sys.maxsize
    best_fun_vals_list = None

    min_fun_vals_list = []
    for i in range(recluster_iters):
        print('--- Iteration %d ---' % (i,))
        lat_clusters = [[] for _ in range(num_lat_clusters)]
        for j, ((x, y), r, lat_cluster_id) in enumerate(circles_copy):
            lat_clusters[lat_cluster_id].append(circles_copy[j])

        min_fun_vals_list_prev = min_fun_vals_list
        min_fun_vals_list = []
        for j, lat_cluster in enumerate(lat_clusters):
            print(lat_cluster)
            # Assume all clusters have been assigned some circles at the beginning
            if not lat_cluster and i > 0:
                # If no circles in cluster, i.e. the circles were all stolen away
                # Use the most recent value for p, the lateration cluster centre.
                # TODO: "transfer" empty clusters to other circle centers far away from the rest.
                min_fun_vals_list.append(min_fun_vals_list_prev[j])
                continue

            min_fun_vals = multilat(lat_cluster, use_local_lims=i>=max(3, recluster_iters/4))
            min_fun_vals['index'] = j
            min_fun_vals_list.append(min_fun_vals)

        total_loss = 0
        for j, min_fun_vals in enumerate(min_fun_vals_list):
            total_loss += min_fun_vals['loss']

        if total_loss < best_total_loss:
            best_total_loss = total_loss
            best_fun_vals_list = deepcopy(min_fun_vals_list)
        # circles_list: list of lateration clusters with circles_copy in them
        print('[multiple_multilateration] min_fun_vals_list', min_fun_vals_list)
        plot_circles(circles_copy, min_fun_vals_list, xlim=xlim, ylim=ylim, iter=i, clear_dir_on_new=False)
        reassign_circle_clusters(circles_copy, min_fun_vals_list)

    # print(circles_copy)
    # print(min_fun_vals_list)

    print(circles_copy)

    # --- Does not work well ---
    """
    # Reduced-radius corrections
    for min_fun_vals in best_fun_vals_list:
        if len(min_fun_vals['circles']) == 3:
            # Make a copy of the circles, then reduce the radii of the copies
            circles = deepcopy(min_fun_vals['circles'])
            circles = [[[x, y], r/1.25, lat_cluster_id] for (x, y), r, lat_cluster_id in circles]

            # Get multilateration results, as well as the loss function.
            # In particular the circle center is of importance.
            reduced_radius_fun_vals = multilat(circles)
            p_from_reduced = reduced_radius_fun_vals['p']
            p_original = min_fun_vals['p']

            # get n-norm loss function with respect to original circles
            loss_func_ord, _, _ = opt_func_dec(*zip(*min_fun_vals['circles']))

            # if the n-norm of lat cluster center and reduced radii **applied
            # to the original circles** is better then the n-norm of the
            # lat cluster center original radii applied to original citcle centers,
            # use this new p_from_reduced
            if loss_func_ord(p_from_reduced, norm_ord=16) < loss_func_ord(p_original, norm_ord=16):
                min_fun_vals['p'] = p_from_reduced
            # else, min_fun_vals['p'] = p_original (which have not changed)

    """

    return best_fun_vals_list


if __name__ == '__main__':
    xlim = (0, 35)
    ylim = (0, 35)
    circles_orig = [
        [[6, 27], 3, None],
        [[3, 25], 2, None],
        [[0, 30], 4, None],

        [[27, 27], 3, None],
        [[26, 27], 2.15, None],
        [[24, 30], 4, None],
        [[22, 22], 4, None],

        [[28, 4], 1.5, None],
        [[26, 3], 2, None],
        [[25.77, 6.8], 2.7, None],
        # [[25.77, 5.5], 2.755, None],

        [[3, 3], 1.5, None],
        [[4.5, 3.5], 2, None],

        [[19, 11], 1.5, None],
        [[20, 14], 2, None],
    ]

    # circles_orig = [
    #     [[15, 15], 6.5, None],
    #     [[12, 15], 2, None],
    #     [[17, 15], 3, None],
    #     [[24, 17], 3.5, None],

    #     # [[8, 6], 1, None],
    #     # [[7.5, 6], 1.5, None],
    # ]

    circles_orig = [
        [[6, 27], 3, None],
        [[3, 25], 2, None],
        [[0, 30], 4, None],

        [[9, 27], 3, None],
        [[6, 25], 2, None],
        [[3, 30], 4, None],
    ]


    num_lat_clusters = 2
    for i, circle in enumerate(circles_orig):
        circle[2] = (i+2) % num_lat_clusters

    best_fun_vals_list = multiple_multilateration(circles_orig, xlim=xlim, ylim=ylim, num_lat_clusters=num_lat_clusters,
                                                  opt_iters=15, recluster_iters=12)

    plot_circles(circles_orig, best_fun_vals_list, xlim=xlim, ylim=ylim)
    print(best_fun_vals_list)