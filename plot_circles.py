# Important: For clearcable project, do not use this file.
# If plots are needed, use either GeoJSON or Localization/plot_circles.py

import os
import glob
import matplotlib.pyplot as plt


def clear_dir(dir):
    files = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), dir, '*.*'))
    images = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), dir, '*.png'))
    assert len(files) == len(images)

    for image in images:
        # print(image)
        os.remove(image)

def plot_circles(circles, min_fun_vals, xlim=(0, 10), ylim=(0, 10), 
                 savefolder=os.path.join('circles', 'results'), iteration=None,
                 clear_dir_on_new=False, title=None, highlight_radius=0.2,
                 labels=None):

    if clear_dir_on_new:
        clear_dir(savefolder)

    circle_plots = []

    for i, circle in enumerate(circles):
        if labels is not None:
            # not yet used
            label = labels[i]
        x, y = circle[0]
        r = circle[1]
        circle_plot = plt.Circle((x, y), r, color='r', fill=False)
        circle_plots.append(circle_plot)

    if min_fun_vals is not None:
        for min_fun_val in min_fun_vals:
            if isinstance(min_fun_val['p'], list):
                for p in min_fun_val['p']:
                    circle_plots.append(
                        plt.Circle(p, highlight_radius, color='purple', fill=False)
                    )
            else:
                circle_plots.append(
                    plt.Circle(min_fun_val['p'], highlight_radius, color='blue', fill=False)
                )

    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for circle_plot in circle_plots:
        ax.add_artist(circle_plot)

    ax.set_aspect('equal')

    if title is None:
        if iteration is None:
            title = 'plotcircles.png'
        else:
            title = 'plotcircles iter %d.png' % (iteration,)

    fig.savefig(os.path.join(savefolder, title))