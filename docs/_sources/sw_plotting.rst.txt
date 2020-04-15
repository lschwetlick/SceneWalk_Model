.. _sw_plotting:

Plotting
========
This module contains functions that help visualize the scenewalk model.

The sw_plot module is used for making dynamical video files.

The nb_plot module are visualization functions that are useful for taking a quick look at data or settings before running estimations.


Example: Dynamic Visualization
==============================
The scenewalk model evolves continuously over time and therefore can be visualized as a video. Since we are visualizing all the in between steps this code is separate from the core model code however and needs to be changed separately in some places when new mechanisms are added.


.. image:: demo/extended_sw.mp4
        :width: 500

.. toctree::
        :caption: Dynamic plotting Notebook
        :maxdepth: 1

        demo/dynamic_plot

Full Plotting Doc
===================
.. automodule:: scenewalk.plotting.sw_plot
        :members:

.. automodule:: scenewalk.plotting.nb_plots
        :members:
