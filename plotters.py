# -*- coding: utf-8 -*-
"""
Plotting tool for Optimizer Analysis -- pulled from PySwarms

This module is built on top of :code:`matplotlib` to render quick and easy
plots for your optimizer. It can plot the best cost for each iteration, and
show animations of the particles in 2-D and 3-D space. Furthermore, because
it has :code:`matplotlib` running under the hood, the plots are easily
customizable.

For example, if we want to plot the cost, simply run the optimizer, get the
cost history from the optimizer instance, and pass it to the
:code:`plot_cost_history()` method

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions.single_obj import sphere
    from pyswarms.utils.plotters import plot_cost_history

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Obtain cost history from optimizer instance
    cost_history = optimizer.cost_history

    # Plot!
    plot_cost_history(cost_history)
    plt.show()

In case you want to plot the particle movement, it is important that either
one of the :code:`matplotlib` animation :code:`Writers` is installed. These
doesn't come out of the box for :code:`pyswarms`, and must be installed
separately. For example, in a Linux or Windows distribution, you can install
:code:`ffmpeg` as

    >>> conda install -c conda-forge ffmpeg

Now, if you want to plot your particles in a 2-D environment, simply pass
the position history of your swarm (obtainable from swarm instance):


.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions.single_obj import sphere
    from pyswarms.utils.plotters import plot_cost_history

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Obtain pos history from optimizer instance
    pos_history = optimizer.pos_history

    # Plot!
    plot_contour(pos_history)

You can also supply various arguments in this method: the indices of the
specific dimensions to be used, the limits of the axes, and the interval/
speed of animation.
"""

# Import standard library
import logging

# Import modules
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
from pyswarms.utils.plotters.formatters import Mesher, Designer, Animator
from pyswarms.utils.plotters.plotters import _mesh

import pandas as pd

    
# NEW STUFF not in ljvmiranda921 pyswarms plotters.py

def plot_summary(
    optimizers,
    canvas=None,
    title="Trajectory",
    titles=None,
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
    n_processes=None,
    **kwargs
):
    """Draw a 2D contour map for particle trajectories

    Here, the space is represented as a flat plane. The contours indicate the
    elevation with respect to the objective function. This works best with
    2-dimensional swarms with their fitness in z-space.

    Parameters
    ----------
    optimizers : numpy.ndarray or list
        List of optimizations to summarize
        :code:`(iteration, n_particles, dimensions)`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking
        the optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    n_processes : int
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :obj:`matplotlib.axes.Axes` plotting function

    Returns
    -------
    :obj:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """

    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(
                limits=[(-1, 1), (-1, 1)], label=["x-axis", "y-axis"]
            )

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig, ax = plt.subplots(3, len(optimizers), figsize=designer.figsize)
        else:
            fig, ax = canvas

        frame_text = ax[2,0].text(0.05, 0.95, s='', transform=ax[0,0].transAxes, horizontalalignment='left',verticalalignment='top')

        # Get number of iterations
        n_iters = len(optimizers[0].record_value['X'])

        # Customize plot
        fig.suptitle(title, fontsize=designer.title_fontsize)

        

        pos_histories = []
        plots = []

        for i, opt in enumerate(optimizers):
            #assert len(opt.record_value['X']) == len(optimizers[0].record_value['X'])
            if titles : ax[0,i].set_title(titles[i])

            Y_history = pd.DataFrame(np.array(opt.record_value['Y']).reshape((-1, opt.size_pop)))
            ax[1,i].set_title(str(opt.gbest_y)+' @ X: '+str(opt.gbest_x), fontsize=8)
            ax[0,i].plot(Y_history.index, Y_history.values, '.')
            Y_history.min(axis=1).cummin().plot(kind='line', ax=ax[1,i])

            ax[2,i].set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
            ax[2,i].set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
            ax[2,i].set_xlim(designer.limits[0])
            ax[2,i].set_ylim(designer.limits[1])

            # Make a contour map if possible
            if mesher is not None:
                (xx, yy, zz) = _mesh(mesher, n_processes=n_processes)
                ax[2,i].contour(xx, yy, zz, levels=mesher.levels)

            # Mark global best if possible
            if mark is not None:
                ax[2,i].scatter(mark[0], mark[1], color="red", marker="x")

            # Put scatter skeleton
            plots.append(ax[2,i].scatter(x=[], y=[], c="black", alpha=0.6, **kwargs))
            #np.expand_dims(pos_histories, 3, np.asarray(opt.record_value['X']))
            pos_histories.append(np.asarray(opt.record_value['X']))


        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate_summary,
            frames=range(n_iters),
            fargs=(pos_histories, plots),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        print("Please check your input type")
        #rep.logger.exception("Please check your input type")
        raise
    else:
        return anim

    

#override of ljvmiranda921 pyswarms
def _animate(i, data, plot):
    """Helper animation function that is called sequentially
    :class:`matplotlib.animation.FuncAnimation`
    """
    current_pos = data[i]
    if i % 10 == 0:
        plot.axes.texts[0].set_text(str(i))

    if np.array(current_pos).shape[1] == 2:
        plot.set_offsets(current_pos)
    else:
        plot._offsets3d = current_pos.T
    return (plot,)

def _animate_summary(i, data, plots):
    """Helper animation function that is called sequentially
    IT ACTUALLY WORKS
    :class:`matplotlib.animation.FuncAnimation`
    """
    if i % 10 == 0:
            plots[0].axes.texts[0].set_text(str(i))

    for j, plot in enumerate(plots):
        opt_data = data[j]
        current_pos = opt_data[i] if i < len(opt_data) else opt_data[-1]

        if np.array(current_pos).shape[1] == 2:
            plot.set_offsets(current_pos)
        else:
            plot._offsets3d = current_pos.T
    return (plots,)

