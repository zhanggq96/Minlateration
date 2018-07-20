# https://stackoverflow.com/questions/17009774/quadratic-program-qp-solver-that-only-depends-on-numpy-scipy

import sys
import numpy as np
import scipy

from copy import deepcopy
from scipy import optimize as opt
from scipy.cluster import hierarchy as hcluster

try:
    # from same directory
    from plot_circles import plot_circles
    from circle_intersection import get_circle_intersections
except ModuleNotFoundError:
    # if it is contained in a project
    from .plot_circles import plot_circles
    from .circle_intersection import get_circle_intersections


# def coord_to_np(coord):
#     return np.array(coord[0]).T
#
def pair_to_np(pair):
    return np.array((pair[0], pair[1])).T

def single_loss(p, x, r):
    return np.abs(np.linalg.norm(p-x) - r)

def single_loss_ord(p, x, r, norm_ord=2):
    return np.abs(np.linalg.norm(p-x, ord=norm_ord) - r) ** norm_ord

def opt_func_dec(x_list, r_list, lat_id_list):
    # x_list: a vector of x's
    # x represents [x y]' : a vector of positions
    # r_list: a vector of r's
    # r represents r: radius of circle corresponding to position

    x_list_copy = deepcopy(x_list)
    r_list_copy = deepcopy(r_list)

    def loss_func_ord(p, norm_ord=2):
        l = sum(
            (single_loss_ord(p, x, r, norm_ord=norm_ord) for x, r in zip(x_list_copy, r_list_copy))
        )

        return l
    # https://stackoverflow.com/questions/4582521/python-creating-dynamic-functions
    def loss_func(p):
        # | ||p-x}| - r |^2
        l = sum(
            (single_loss(p, x, r) for x, r in zip(x_list_copy, r_list_copy))
        )

        return l

    def jacobian(p):
        grad_j = np.zeros_like(p)
        for xi, ri in zip(x_list_copy, r_list_copy):
            d = np.linalg.norm(p-xi)
            grad_j += 2*(d-ri) * (1/2)*(1/d) * 2*(p-xi)
        return grad_j

    return loss_func_ord, loss_func, jacobian


def determine_num_lat_clusters(circles, clustering_threshold=0.2):

    def perform_hcluster(points, clustering_threshold=0.2, metric='euclidean'):
        # points: column np array of [x; y] containing intersections and circle centers
        # do hcluster.
        enum_clusters = hcluster.fclusterdata(points, clustering_threshold, criterion='distance',
                                              metric=metric)
        # enumerate the clusters from zero to # clusters-1
        enum_clusters -= 1
        num_lat_clusters = np.max(enum_clusters) + 1
        # print('[perform_hcluster] num_lat_clusters:', num_lat_clusters)

        lat_clusters = [[] for _ in range(num_lat_clusters)]
        for point, point_cluster in zip(points, enum_clusters):
            lat_clusters[point_cluster].append(point)

        # Calculate the cluster means
        cluster_means = []
        for cluster in lat_clusters:
            mean = np.zeros_like(cluster[0], dtype='float64')
            for point in cluster:
                mean += point
            mean /= len(cluster)
            cluster_means.append(mean)

        # return the number of clusters and the clusters enumerated from zero
        return num_lat_clusters, enum_clusters, cluster_means

    # Initiallize list of points to be used in hcluster from scikit.
    # They will include circle radii and intersection points between
    # all pairs of circles.
    hcluster_points = []

    # list of indices of the centers of each circle, in the order
    # in which they are appended to hcluster points
    circle_point_id_list = []
    point_count = 0

    # iterate over all pairs
    for i, circle0 in enumerate(circles):
        c0_origin = circle0[0]

        # get circle origins, and add them to hcluster points
        hcluster_points.append(c0_origin)

        circle_point_id_list.append(point_count)
        point_count += 1

        for j, circle1 in enumerate(circles[i+1:], i+1):

            # Determine whether the circles actually intersect or not;
            # if so, return intersection points as 2 column numpy vectors
            # - In the case of one tangential intersection, return same point
            #   twice
            # - In the case of containment, ix0 and ix1's truthfulness determines
            #   which circle is the larger (containing) one.
            ix0, ix1, case = get_circle_intersections(circle0, circle1)

            # if case in {'seperate', 'contained', 'coincident'}:
            # the circles could still be close enough together.
            # so try increasing/decreasing their radii to see if we
            # can detect an "almost-intersection".
            if case == 'seperate':
                # seperate: try increasing radii and detecting intersection
                c0, r0, lat_cluster0 = circle0
                c1, r1, lat_cluster1 = circle1

                # expand both radii
                circle0_expanded = [c0, 1.1 * r0, lat_cluster0]
                circle1_expanded = [c1, 1.1 * r1, lat_cluster1]
                ix0, ix1, case = get_circle_intersections(circle0_expanded, circle1_expanded)
                if case == 'intersect':
                    hcluster_points.extend([ix0, ix1])
                    point_count += 2
            elif case == 'contained':
                # seperate: try decreasing radii and detecting intersection
                c0, r0, lat_cluster0 = circle0
                c1, r1, lat_cluster1 = circle1

                # reduce the outer (containing) circle's radius
                # expand the inner (contained) circle's radius
                circle0_reduced = [c0, (0.9 if ix0 is True else 1.1) * r0, lat_cluster0]
                circle1_reduced = [c1, (0.9 if ix1 is True else 1.1) * r1, lat_cluster1]
                ix0, ix1, case = get_circle_intersections(circle0_reduced, circle1_reduced)
                if case == 'intersect':
                    hcluster_points.extend([ix0, ix1])
                    point_count += 2
            elif case == 'intersect':
                # if they already intersect, great!
                hcluster_points.extend([ix0, ix1])
                point_count += 2

    num_lat_clusters, enum_clusters, cluster_means = \
        perform_hcluster(hcluster_points, clustering_threshold=clustering_threshold)
    return num_lat_clusters, enum_clusters, cluster_means, circle_point_id_list

def get_local_lims(circles):
    max_x = -sys.maxsize
    min_x = +sys.maxsize
    max_y = -sys.maxsize
    min_y = +sys.maxsize
    for (x, y), r, lat_cluster_id in circles:
        max_x = max(x + r, max_x)
        min_x = min(x - r, min_x)
        max_y = max(y + r, max_y)
        min_y = min(y - r, min_y)

    return (min_x, max_x), (min_y, max_y)

def multiple_multilateration(circles_ref, xlim=(0,10), ylim=(0,10),
                             num_lat_clusters=2, opt_trials=7, recluster_iters=5,
                             clustering_threshold=4.5, highlight_radius=0.2,
                             verbose=False):

    circles_copy = deepcopy(circles_ref)
    num_circles = len(circles_copy)
    if verbose: print('[multiple_multilateration] circles_copy init:', circles_copy)

    options = {'disp': False}

    # ------------------- Begin Helper Functions -------------------

    def multilat(circles, use_local_lims=False, p0_from_hcluster=None):

        loss_func_ord, loss_func, grad_j = opt_func_dec(*zip(*circles))

        min_fun_vals = {
            'loss': sys.maxsize,
            'p': None,
            'circles': deepcopy(circles),
            'index': None
        }

        # Get coordinate limits of the cluster's data
        if use_local_lims:
            cluster_xlim, cluster_ylim = get_local_lims(circles)
        else:
            cluster_xlim, cluster_ylim = xlim, ylim

        for ot in range(opt_trials):
            # p0 = np.array([7, 7]).T
            # p0 = np.random.uniform(0, 30, p0_example.shape)

            # Generate single random initial cluster center
            # always do so for the initial try
            # if p0_from_hcluster is not None and (ot == 0 or ot % 2 == 0):
            if p0_from_hcluster is not None and (ot == 0):
                p0 = p0_from_hcluster
            else:
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

        for i, (center, r, lat_cluster_id) in enumerate(circles):

            x = center
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

    # ------------------- End Helper Functions -------------------

    # ------------------- Determine initial points if appropriate -------------------

    # List of initial circle intersection guesses
    p0_list = None

    if num_lat_clusters is None:
        num_lat_clusters, enum_clusters, cluster_means, circle_point_id_list \
            = determine_num_lat_clusters(circles_copy, clustering_threshold=clustering_threshold)

        # iterate over all points used in hcluster *which are circles*
        # the indices of such points were returned in circle_point_id_list
        for circle_i, circle_point_id in zip(range(num_circles), circle_point_id_list):
            # get the cluster which circle i is in
            circle_cluster_id = enum_clusters[circle_point_id]

            # set the init clusters for each circle
            circles_copy[circle_i][2] = circle_cluster_id

        p0_list = cluster_means

    # ------------------- End determine initial points if appropriate -------------------

    best_total_loss = sys.maxsize
    best_fun_vals_list = None

    min_fun_vals_list = []
    for i in range(recluster_iters):
        if verbose: print('--- Iteration %d ---' % (i,))
        lat_clusters = [[] for _ in range(num_lat_clusters)]
        for j, ((x, y), r, lat_cluster_id) in enumerate(circles_copy):
            lat_clusters[lat_cluster_id].append(circles_copy[j])

        min_fun_vals_list_prev = min_fun_vals_list
        min_fun_vals_list = []
        for j, lat_cluster in enumerate(lat_clusters):
            if verbose: print('Cluster # %d:' % (j,), lat_cluster)
            # Assume all clusters have been assigned some circles at the beginning
            if not lat_cluster and i > 0:
                # If no circles in cluster, i.e. the circles were all stolen away
                # Use the most recent value for p, the lateration cluster centre.

                # TODO: "transfer" empty clusters to other circle centers far away from the rest.
                # Addendum: I probably won't bother adding this unless it seems like it can
                # significantly improve performance. I'll leave the argmax_x function
                # here for now if it seems I will use distant circle transfer in the future.
                min_fun_vals_list.append(min_fun_vals_list_prev[j])
                continue
            elif not lat_cluster and i == 0:
                # if, on initialization, there was a cluster with no circles assigned to it
                min_fun_vals = {
                    'loss': sys.maxsize,    # Essentially infinite loss
                    'p': p0_list[j],        # Use initial values of p from hierarchical (avg of points)
                    'circles': [],          # Circles are usually stored in lat_cluster.
                                            # However, is an empty list so deepcopy(lat_cluster) is [].
                    'index': j              # Cluster # j
                }
                min_fun_vals_list.append(min_fun_vals)
                continue

            # perform multilateration.
            # use initial points corresponding to cluster j for the initial try of optimization;
            # the initial try is only the first iteration of reclustering (i == 0)
            min_fun_vals = multilat(lat_cluster, use_local_lims=i>=max(2, recluster_iters/4),
                                    p0_from_hcluster=p0_list[j] if i == 0 else None)
            min_fun_vals['index'] = j
            min_fun_vals_list.append(min_fun_vals)

        total_loss = 0
        for j, min_fun_vals in enumerate(min_fun_vals_list):
            total_loss += min_fun_vals['loss']

        if total_loss < best_total_loss:
            best_total_loss = total_loss
            best_fun_vals_list = deepcopy(min_fun_vals_list)

        if verbose: print('min_fun_vals_list', min_fun_vals_list)
        plot_circles(circles_copy, min_fun_vals_list, xlim=xlim, ylim=ylim, 
                     iteration=i, clear_dir_on_new=False, highlight_radius=highlight_radius)
        reassign_circle_clusters(circles_copy, min_fun_vals_list)

    # if verbose:
    #     print(circles_copy)

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
    if best_fun_vals_list is None:
        best_fun_vals_list = min_fun_vals_list

    for best_fun_vals in best_fun_vals_list:
        if len(best_fun_vals['circles']) == 2:
            ix0, ix1, case = get_circle_intersections(*best_fun_vals['circles'])
            if case == 'intersect':
                # print('intersections:', ix0, ix1)
                best_fun_vals['p'] = [ix0, ix1]

    return best_fun_vals_list, best_total_loss

def locate_intersections(circles_ref, xlim=None, ylim=None, num_lat_clusters=None, clustering_threshold=None,
                         verbose=False):

    assert circles_ref
    assert (xlim is not None and ylim is not None) or (xlim is None and ylim is None)

    r_avg = 0
    circles_np = []
    for circle in circles_ref:
        x, y = circle[0]
        r = circle[1]
        circles_np.append([pair_to_np([x, y]), r, None])
        r_avg += r
    r_avg /= len(circles_np)
    highlight_radius = r_avg / 10

    if xlim is None or ylim is None:
        xlim, ylim = get_local_lims(circles_np)
        if verbose: print('[minlateration: locate_intersections] xlim, ylim:', xlim, ylim)

    if clustering_threshold is None:
        K = 34
        diameter = max(xlim[1]-xlim[0], ylim[1]-ylim[0])
        # threshold: determine based on diamater of area covered
        # and circle average sizes
        # (determined these by trial and error)
        if diameter / (r_avg*3) > 4:
            clustering_threshold = (diameter / K) * diameter / (r_avg*3)
        elif diameter / (r_avg*3) > 2:
            clustering_threshold = (diameter / K) * 4
        else:
            clustering_threshold = ((diameter / K) * diameter / (r_avg*3)) ** 2

        if verbose: print('[minlateration: locate_intersections] auto cluster threshold: %f'
                          % (clustering_threshold,))

    best_fun_vals_list, best_total_loss = \
        multiple_multilateration(circles_np, xlim=xlim, ylim=ylim, num_lat_clusters=num_lat_clusters,
                                 opt_trials=15, recluster_iters=5, clustering_threshold=clustering_threshold,
                                 highlight_radius=highlight_radius, verbose=verbose)

    if verbose: print('Best total loss:', best_total_loss)

    return best_fun_vals_list, best_total_loss, xlim, ylim, highlight_radius

if __name__ == '__main__':
    xlim = None
    ylim = None

    # Test case 0:
    circles_ref = [
        [[6, 27], 3],
        [[3, 25], 2],
        [[0, 30], 4],

        [[27, 27], 3],
        [[26, 27], 2.15],
        [[24, 30], 4],
        [[22, 22], 4],

        [[28, 4], 1.5],
        [[26, 3], 2],
        [[25.77, 6.8], 2.7],
        # [[25.77, 5.5], 2.755],

        [[3, 3], 1.5],
        [[4.5, 3.5], 2],

        [[19, 11], 1.5],
        [[20, 14], 2],
    ]

    # Test case 1:
    # circles_ref = [
    #     [[3, 4], 1.2, 0],
    #     [[3, 7], 3, 0],
    #     [[5, 4], 1, 0],
    #     [[0, 0], 5.66, 0],
    #     [[7.5, 6], 2, 1],
    #     [[8, 6], 1.5, 1],
    # ]

    # Test case 2:
    # circles_ref = [
    #     [[15, 15], 6.5, None],
    #     [[12, 15], 2, None],
    #     [[17, 15], 3, None],
    #     [[24, 17], 3.5, None],
    #
    #     [[8, 6], 1, None],
    #     [[7.5, 6], 1.5, None],
    # ]

    # Test case 3:
    # circles_ref = [
    #     [[6, 27], 3, None],
    #     [[3, 25], 2, None],
    #     [[0, 30], 4, None],
    #
    #     [[9, 27], 3, None],
    #     [[6, 25], 2, None],
    #     [[3, 30], 4, None],
    # ]

    # plot_circles(circles_ref, None, xlim=xlim, ylim=ylim, title='plotcircles init')

    best_fun_vals_list, best_total_loss, xlim, ylim, highlight_radius = \
        locate_intersections(circles_ref, xlim=xlim, ylim=ylim, num_lat_clusters=None,
                             clustering_threshold=None, verbose=True)

    plot_circles(circles_ref, best_fun_vals_list, xlim=xlim, ylim=ylim, highlight_radius=highlight_radius)
    # print(best_fun_vals_list)