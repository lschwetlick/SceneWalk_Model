# SceneWalk Model
## TODO
- config file for data path
- fix pydream to have it as a dependency
- benchmarking the model?

## Getting Started
1. activate your virtual environment and/or make one
```
cd ~/Documents/virtual_envs
python3 -m venv sw_env
source sw_env/bin/activate
```
2. install scenewalk package
```
cd ~/Scenewalk
pip3 install -e .
```
3. install the venv as a kernel for notebooks
```
pip3 install ipykernel
python3 -m ipykernel install --name sw_env
```
4. add the location of your Data to the config file. Simply go to the `config.yml` file and change the paths and/or dataset names!
5. use it like any other package!

## 0. Repo Structure and Modules
### Repo
- `scenewalk`: package containing the scenewalk code. More documentation on this below.
- `tests`: contains the test suite for the model and the model interfaces
- `demo`: some visualizations and examples
### Core
- `scenewalk_model_object.py`: Model Code
- `scenewalk_limited`: [experimental] variation of the SceneWalk model that uses only limited number of past fixations
- `scenewalk_missed_target`: [experimental] variation of the SceneWalk model with slightly different assumptions
### Interfaces (submodules of `scenewalk`)
- `estimation`: contains code for passing the model and priors into PyDream
- `evaluation`: contains code for parallelizing the model evaluation over subjects and trials
- `simulation`: contains code for simulating data sets
- `utils`: useful functions when working with the model
- `plotting`: frequently used visualization functions

## 1. Model Code
### The Scenewalk Object
The `scenewalk_model_object` file defines a python object which can be instantiated using different settings. The currently available settings are
- inhib_method: divisive or subtractive inhibition
- att_map_init_type: is there a center bias on the initial attention map
- postsaccadic_shift: enables postsaccadic shift. WARNING: This implementation is broken and needs fixing!!!!!!
- presaccadic_shift_switch: enables presaccadic shift
- exponents: sets whether lambda and gamma, the two shaping parameters for attention and inhibition are the same
- locdep_decay: sets whether there is location dependent decay (facilitation of return mechanic that makes the fish appear)

Additionally, there are some optional settings that can be passed. These are currently enjoying a test run:
- coupled_oms: changes the omega inhibition to always be a fraction of omega attention (size of fraction is given by "omfrac" parameter)
- coupled_sigmas: sets both sigmas to be equal
- logged_cf: allows us to estimate log(inhib_strength) instead of inhib_strength
- logged_z: allows us to estimate log(zeta) instead of zeta

The code architecture that enables switching between different mechnics being included in the model works like this:
The model's internal funtion names always stay the same, no matter which configuration (because it still does basically the same things, just in a differnt way. E.g. an attention map is always constructed, it is then always evolved.)
In the constructor we set a flag telling the class the model specifications. The instantiation then finds the correct function for this mechanism and assigns it the more abstract, globally valid name. For example we might have two versions of initiating the attention map, with cb and without. There exist two functions that construct attention maps accordingly (init_att_cb and init_map_no_cb). The constructor then takes the version indicated by the flag and assigns it to the handle init_att. The model then uses that handle to adress the function and therefore works with both versions.
This is neat because a) not too much code duplication and b) we can easily separate out one version of the model if we ever need the ONE canonical sw model.

The object can be used to find individual intermediary maps but it has 2 main interface functions:
- get_scanpath_likelihood: gives the likelihood of a scanpath given an image
- simulate_scanpath: simulates a scanpath on the image

## 2. Models
### Core parameters
- omegaAttention: speed of decay of the previous attention map
- omegaInhib: speed of decay of the previous inhibition map
- sigmaAttention: size of the attention gaussian
- sigmaInhib: size of the attention gaussian
- gamma: shaoing of the inhibtion map
- lamb: shaoing of the inhibtion map
- inhibStrength: influence of the inhibition map on the final map
- zeta: noise

**WATCH OUT** with the inputs:
- Time is in seconds, not miliseconds
- Distances refer to degrees of visual angle

### 2.1 Original Model
#### Bounds
Due to restrictions on numerical computations the model is only defined within certain bounds. Using the model with parameters outside of these bounds will cause errors, as values may become small enough to be flushed to zero. At this point the code uses 128 floats. The bounds are of course dependent on the number of bits in the float type.

Specifically **lamb** and **gamma** are problematic. They should operate in Q+ but in due to the limits of computational ressources (see notebook floating point troubles) has an upper bound that is dependent on the float type and on the lower bound of OmegaAttention and the upper bound of SigmaAttention. lambda's upper bound is lower than gamma's, because in the attention pathway the multiplication with the saliency makes all the values smaller!

**zeta**, being a probability needs to be between 0 and 1 to make sense.


Current version of the subtractive model works in the following scope
- **omegaAttention** float64.eps : 100000
- **omegaInhib** float64.eps : 100000
- **sigmaAttention** float64.eps : 100000
- **sigmaInhib** float64.eps : 100000
- **lamb** float64.eps : 17
- **gamma** float64.eps : 17
- **inhibStrength** float64.eps : 100000
- **zeta** float64.eps : 1

Divisive inhibition version works in the following scope
- **omegaAttention** float64.eps : 100000
- **omegaInhib** float64.eps : 100000
- **sigmaAttention** float64.eps : 100000
- **sigmaInhib** float64.eps : 100000
- **lamb** float64.eps : 15
- **gamma** float64.eps : 15
- **inhibStrength** float64.eps : 100000
- **zeta** float64.eps : 1


### 2.2 Preceeding Attention Model
In this model the attention stream precedes the fixation and the inhibition to the next location.
`delay` is the amount of time by which attention precedes inhibition.
In this implementation this works in 3 steps
1. evolve the model according to the original scenewalk model until time (`duration-delay`)
2. pick next fixation location and or evaluate likelihood at this time
3. evolve the model with inhibition still in the old locaton but attention already in the new for `delay` time

Functions unique to this model variant have "_delay" in their name, and should otherwise be analogous to the original model's functions.

### 2.3 Spatial Attention Shift Model
- aka remap model
- aka lars' model

This model accounts for more effects found in empirical data by adding two mechanisms to the model:
1. A center bias to the initialization map and
2. A remapping of the focus of attention at the time of the saccade in the direction of the saccade.

it introduces 6 new paramters:
- sigmaShift: size of the shifted gaussian
- shift_size: how far from the fixation position is the shift
- phi: speed of the decay of the shift
- first_fix_OmegaAttention: speed of decay of the center bias
- cb_sd_x: center bias width in x direction
- cb_sd_y: center bias width in y direction

#### Bounds
Works in the following scope
- **omegaAttention** float64.eps : 100000
- **omegaInhib** float64.eps : 100000
- **sigmaAttention** float64.eps : 100000
- **sigmaInhib** float64.eps : 100000
- **lamb** float64.eps : 15
- **gamma** float64.eps : 15
- **inhibStrength** float64.eps : 100000
- **zeta** float64.eps : 1
- **sigmaShift**: float64.eps : 100000
- **shift_size**: float64.eps : 100000
- **phi**: 1 : 100000
- **first_fix_OmegaAttention**: float64.eps : 100000
- **cb_sd_x**: 0.1 : 100000
- **cb_sd_y**: 0.1 : 100000

## 3. Data
The expected format of the data is...

## 4. Tests
The Test suite checks the following properties of the main (original) model:
- Datatype of likelihood maps is float128
- Likelihood maps are never
    - NaN
- The final map can never be
    - negative
    - zero
    - not a density

## 5. Modules
### 5.1 Evaluation
This module is for paralellizing the call to get scanpath likelihoods over subjects when looking at likelihoods of whole datasets
### 5.2 Evaluation
This module runs a bayesian parameter inference using the DREAM package.
There is code to do this in parallel and seperately for multiple people or to get an average over all people
### 5.3  plotting
This module can make nice videos and some basic stationary plots of maps in the scenewalk model
### 5.4 simulation
This module simulates data sets, normally given the fixation durations and images of a reference data set.