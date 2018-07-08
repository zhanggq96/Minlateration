from plot_circles import plot_circles


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
    circle_sandbox()