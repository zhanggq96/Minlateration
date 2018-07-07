# https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot

import os
import matplotlib.pyplot as plt

def plot_circles(circles, min_fun_vals, savefolder='circles'):

    # circle1 = plt.Circle((0, 0), 0.2, color='r')
    # circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
    # circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)

    circle_plots = []

    for (x, y), r, lat_cluster_id in circles:
        circle_plot = plt.Circle((x, y), r, color='r', fill=False)
        circle_plots.append(circle_plot)

    for min_fun_val in min_fun_vals:
        circle_plots.append(
            plt.Circle(min_fun_val['p'], 0.2, color='blue', fill=False)
        )

    fig, ax = plt.subplots()
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 10))
    for circle_plot in circle_plots:
        ax.add_artist(circle_plot)

    ax.set_aspect('equal')
    fig.savefig(os.path.join(savefolder, 'plotcircles.png'))