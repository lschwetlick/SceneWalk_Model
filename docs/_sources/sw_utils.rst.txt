.. _sw_utils:

Utils
=====
This module is a collection of various useful tools for the use of the scenewalk model.


loadData
========
As decribed in the section "data requirements" the package expects data to be organized as a list of lists of arrays referring to the levels of subject, trial and fixation respectively. The relevant data are x and y coordinates, image number and  fixation duration. Additionally the model needs the fixation densities/saliency maps and the data range.

In loadData datasets are loaded into dictionaries, can be shortened or split in various ways and then saved to their own variables.


.. automodule:: scenewalk.utils.loadData
        :members:

utils
=====
These funtions mainly refer to saving estimation chains returned by pydream  into more readable formats

.. automodule:: scenewalk.utils.utils
        :members:


resort
======
Use this thing with utmost caution. It is written to suit one particular folder structure and file naming conventions and will likely need adapting to  other setups!
This is a little commandline script that sorts the estimation files into folders. Since this depends on the precise setup, and folder structure use it with caution! The idea is to combine error and output files with chains files into separate folders for each subject. Use::

	python3 -m scenewalk.utils.resort "2019" 5 -d -o

where the first argument is the beginning of the id number, the second argument is how many estimations there are, the ``-d`` is the dry run option (without it files will be moved for real) and ``-o`` is a different file name structure.

This will probably not word without modification on other setups!

