.. _data_requirements:

=================
Data Requirements
=================

Whether its used for estimating parameter values or for simulating data, the SceneWalk model expects  data to come in the following structure:


Scan Path Information
---------------------

- x and y positions in degrees of visual angle. The origin of the data is expected to be at the bottom left.
- fixation durations in seconds
- the image number (should start at 1)

These data (x, y, duration, image) should be organized as a list of list of numpy arrays like so: ``Subjects[Trial[Scanpath[]]]``. So each scanpath is a numpy array and then all the scanpaths go into a list of trials for one subject. Lastly all trial lists are added to a list of all subjects.

Densities and Range
-------------------

- the data range in degrees of visual angle as ``[[xmin, xmax],[ymin, ymax]]``
- a list of 'priority maps'. We use empirical fixation density maps but it is possible to use model generated saliency maps. By default the model expects 128x128 densities, but it can be configured to accept different resolutions. The densities should sum to 1 and be 128x128 pixel numpy arrays

Folder Structure
----------------

In the utils.loadData submodule we  assume that the above information has been saved to a folder called ``npy``, where each bullet point is a separate ``.npy`` file, with some name, but ending in "_x.npy", "_y.npy", "_dur.npy", "_im.npy", "_range.npy", and "_densities.npy". 

The folder structure for datasets is therefore as follows::

    |--DATA
    |-- Dataset1
        |-- npy
        |-- d1_x.npy
        |-- d1_y.npy
        |-- d1_im.npy
        |-- d1_dur.npy
        |-- d1_densities.npy
        |-- d1_range.npy
    |-- Dataset2
        |-- npy
        |-- d2_x.npy
        |-- d2_y.npy
        |-- d2_im.npy
        |-- d2_dur.npy
        |-- d2_densities.npy
        |-- d2_range.npy


Example
-------
.. toctree::
        :caption: Core Model Code
        :maxdepth: 1

        demo/get_corpus_data


In the Utils.loadData Module
============================
If you have set up a DATA folder as shown above you can use the inbuilt loading functions to import the data. However you must inform the library where to look for this folder. There are 4 options to tell the module where to look for data:

1. you pass the the path to the ``utils.loadData.load_data()`` function directly. Use an absolute path the to npy folder.
2. you can set the path at the top of  your script like so::

    from scenewalk.utils import loadData
    loadData.DATA_PATH = "My/Path/DATA"

3. you can place a ``config.yml`` file into your working directory (see ``config_sample.yml``)
4. you can place a ``config.yml`` file into the ``scenewalk_model/scenewalk`` directory (see ``config_sample.yml``)

Once this is set up, the data loading should work as shown in the examples of this documentation.

