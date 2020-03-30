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

These data (x, y, duration, image) should be organized as a list of list of np arrays like so: `Subjects[Trial[Scanpath[]]]`. So each scanpath is a numpy array and then all the scanpaths go into a list of trials for one subject. Lastly all trial lists are added to a list of all subjects.

Furthermore we need
-------------------

- the data range in degrees of visual angle
- a list of 'saliency maps'. In the paper we use empirical fixation density maps but it is possible to use model generated saliency maps. By default the model expects 128x128 densities, but it can be configured to accept different resolutions. The densities should sum to 1 and be 128x128 pixel numpy arrays

In the utils.loadData submodule I assume that this information has been saved to a folder, where each bullet point is a separate npy file, with some name, but ending in "_x", "_y", "_dur", "_im", "_range", and "_densities". This the path to this folder is what should be added to the `config.yml` file.
