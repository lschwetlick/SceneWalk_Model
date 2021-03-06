.. SceneWalk documentation master file, created by
   sphinx-quickstart on Fri Mar 27 16:42:30 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SceneWalk Model
===============
This is the documentation for the SceneWalk Model Package.

Here is a link to the paper: https://www.nature.com/articles/s42003-020-01429-8.epdf

and the preprint: https://psyarxiv.com/zcbny

Here is a link to the code needed to exactly reproduce the findings from the paper: https://osf.io/qsx4w/

This documentation is for anyone wanting to use the SceneWalk Model on their own data and for their own research. Some words about the implementation: the idea for the infrastructure is that the model is a python object. It is instantiated with some main settings which define the mechanisms that are switched on and off. In the model extensions we added a variety of mechanisms, each one of which can be used or not in a modular way. Furthermore the model object sets various specifics on how different parameters are represented (i.e. coupled with others or on a log scale.) Lastly, The model must be given parameters. Given all these inputs the model object can be called to output likelihood values given data or to work generatively to simulate scanpaths, For details please consult the following documentation.

The package also contains modules that use the SceneWalk code to do things, such as estimate parameters or plot scan paths.

.. image:: demo/extended_sw.gif
        :width: 600

.. toctree::
        :caption: Using the SceneWalk Package
        :maxdepth: 1

        getting_started
        data_requirements
        demo/how_to_sw

.. toctree::
        :caption: Core Model Code
        :maxdepth: 1        

        demo/detailed_look_at_sw
        sw_core


.. toctree::
        :caption: Evaluation
        :maxdepth: 2

        sw_evaluation

.. toctree::
        :caption: Estimation
        :maxdepth: 2
        
        sw_estimation


.. toctree::
        :caption: Simulation
        :maxdepth: 2

        sw_simulation


.. toctree::
        :caption: Plotting
        :maxdepth: 1

        sw_plotting


.. toctree::
        :caption: Utils
        :maxdepth: 1

        sw_utils


.. toctree::
        :caption: Testing & Benchmarking
        :maxdepth: 1

        testing
	benchmarking

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

