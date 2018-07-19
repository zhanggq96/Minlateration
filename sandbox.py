from plot_circles import plot_circles
from multilateration import *


def circle_sandbox(num_lat_clusters=5, opt_iters=12, recluster_iters=5):
    xlim = (0,30)
    ylim = (0,30)
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
        [[25.77, 5.5], 2.755, None],

        [[3, 3], 1.5, None],
        [[4.5, 3.5], 2, None],

        [[3, 3], 1.5, None],
        [[4.5, 3.5], 2, None],

        [[19, 11], 1.5, None],
        [[20, 14], 2, None],
    ]

    for i, circle in enumerate(circles_orig):
        circle[2] = i % num_lat_clusters

    plot_circles(circles_orig, None, xlim=xlim, ylim=ylim)

if __name__ == '__main__':
    # circle_sandbox()
    # circles_orig = [
    #     [[6, 27], 3, None],
    #     [[3, 25], 2, None],
    #     [[0, 30], 4, None],
    # ]
    # for i, circle in enumerate(circles_orig):
    #     for j, circle2 in enumerate(circles_orig[i+1:], i+1):
    #         print(i, j, circle, circle2)

    # circles_ref = [
    #     [[6, 27], 3, None],
    #     [[3, 25], 2, None],
    #     [[0, 30], 4, None],
    #
    #     [[27, 27], 3, None],
    #     [[26, 27], 2.15, None],
    #     [[24, 30], 4, None],
    #     [[22, 22], 4, None],
    #
    #     [[28, 4], 1.5, None],
    #     [[26, 3], 2, None],
    #     [[25.77, 6.8], 2.7, None],
    #     # [[25.77, 5.5], 2.755, None],
    #
    #     [[3, 3], 1.5, None],
    #     [[4.5, 3.5], 2, None],
    #
    #     [[19, 11], 1.5, None],
    #     [[20, 14], 2, None],
    # ]
    #
    # num_circles = len(circles_ref)
    #
    # num_lat_clusters, enum_clusters, circle_point_id_list = \
    #     determine_num_lat_clusters(circles_ref, clustering_threshold=5)
    # # enum_clusters: which cluster each intersection is assigned to
    # print(num_lat_clusters)
    # print(enum_clusters)
    #
    # # def filter_gen(num_lat_clusters):
    # #     filter_funcs = []
    # #     for i in range(num_lat_clusters):
    # #         filter_funcs.append(lambda k, t=i: enum_clusters[k] == t)
    # #
    # #     return filter_funcs
    # # for i in range(num_lat_clusters):
    # #     # cluster_points = filter(lambda (k, _): i == k , (range(num_circles), circles_ref))
    # #     cluster_points = filter(lambda range_circle: i == range_circle[0], zip(range(num_circles), circles_ref))
    # # filter_funcs = filter_gen(num_lat_clusters)
    # # for filter_func in filter_funcs:
    # #     x = list(filter(filter_func, circles_ref)
    # #     print(x)
    #
    # # initial empty clusters. Will add each circle to a cluster
    # lat_clusters = [[] for _ in range(num_lat_clusters)]
    #
    # # iterate over all points used in hcluster *which are circles*
    # # the indices of such points were returned in circle_point_id_list
    # for circle_i, circle_point_id, circle in zip(range(num_circles), circle_point_id_list, circles_ref):
    #     # get the cluster which circle i is in
    #     circle_cluster_id = enum_clusters[circle_point_id]
    #     print('Circle %d\'s cluster: %d' % (circle_i, circle_cluster_id))
    #     # add this circle to appropriate cluster in lat_clusters
    #     lat_clusters[circle_cluster_id].append(circle)
    #
    # cluster_means = []
    # for cluster in lat_clusters:
    #     mean_x, mean_y = 0, 0
    #     for (x, y), r, _ in cluster:
    #         mean_x += x
    #         mean_y += y
    #     mean_x /= len(cluster)
    #     mean_y /= len(cluster)
    #     cluster_means.append([mean_x, mean_y])
    #
    # print(cluster_means)

    x = np.array([1, 2])
    y = np.array([1, 2])
    z = [x, y]

    for a, b in z:
        print(a, b)
