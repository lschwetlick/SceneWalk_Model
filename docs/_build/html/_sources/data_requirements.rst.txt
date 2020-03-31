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

Folder Structure
----------------

In the utils.loadData submodule I assume that this information has been saved to a folder, where each bullet point is a separate npy file, with some name, but ending in "_x", "_y", "_dur", "_im", "_range", and "_densities". 

The folder structure for datasets is therefore as follows:

```
|--DATA
  |-- Dataset1
    |-- npy
      |-- d1_x.npy
      |-- d1_y.npy
      |-- d1_im.npy
      |-- d1_dur.npy
      |-- d1_densities.npy
      |-- d1_densities.npy
  |-- Dataset2
    |-- npy
      |-- d2_x.npy
      |-- d2_y.npy
      |-- d2_im.npy
      |-- d2_dur.npy
      |-- d2_densities.npy
      |-- d2_densities.npy 
```

You have multiple options to tell the module where to look for data:
1. you pass the the path to the load_data() function directly. Use an absolute path the to npy folder.
2. you can set the path in your script like so:
```
from scenewalk.utils import loadData
loadData.DATA_PATH = "My/Path/DATA"
```
3. you can place a config.yml file into your working directory (see config_sample.yml)
4. you can place a config.yml file into the scenewalk_model top level directory (see config_sample.yml)

