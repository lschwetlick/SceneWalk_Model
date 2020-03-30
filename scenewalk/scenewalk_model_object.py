"""
Scenewalk Model Implementation

Lisa Schwetlick 2019

University of Potsdam
"""
import warnings
from collections import OrderedDict
import numpy as np

class scenewalk:
    """
    Model class. Makes a scenewalk model according to the configurations given
    at instantiation (see below). Some of the most important features for the
    workflow are:
        - `whoami()` displays a string describing the configuration.
        - `get_param_list_order()` displays the exposed parameters that result \
        from the chosen configuration.
        - `update_params()` lets you set parameter values.
        - `get_scanpath_likelihood` returns a scan path likelihood given the \
        model.
        - `simulate_scanpath` returns a scan path simulated by the model.

    Notes
    -----
        - time data is always given in seconds (not ms)
        - position data is always given in degrees (not pixels)
        - the model is tested mainly on an internal representation of 128x128\
        pixels. Using `set_mapsize()` this can be changed.
        - the model uses an internal representation of np.float128. This is\
        because for estimation, parameters can have "extreme" values that make\
        the model numerically instable due to floating point problems. It also\
        makes the computation slower, of course. Using `set_precision()` this\
        can be changed.

    Parameters
    ----------
    inhib_method : {'subtractive', 'divisive'}
        How are attention and inhibition map combined?
    att_map_init_type : {'zero', 'cb'}
        How is the attention map initialized?
    postsaccadic_shift_switch : {'on', 'off'}
        is there a postaccadic attention shift?
    presaccadic_shift_switch : {'on', 'off'}
        is there a presaccadic attention shift?
    locdep_decay_switch : {'on', 'off'}
        Is there slower decay on he previous fixation location?
    omp : {"off", add", "mult"]}
        Adds occulomotor potential
    data_range : dict
        dictionary ({'x' : [0,127], 'y' : [0,127]}) of the data range.
    kwargs_dict : dict
        any other model attributes you may want to set. They pertain mainly to
        how the parameters of the model are exposed in set_params. For Example
            - exponents [1 or 2]: are Lambda and Gamma independent?
            - coupled_oms [True, False]: True changes the parametrization of \
                omega to have om_i = om_a / om_frac
            - coupled_sigmas [True, False]: True changes the parametrization of\
                sigma to be sig_i = sig_a
            - logged_cf [True, False]: True makes the cf = 10**input_cf
            - logged_z [True, False]: True makes the zeta = 10**input_zeta
            - logged_ompf [True, False]: True makes the ompf = 10**input_ompf
            - coupled_facil [True, False]: True makes \
                omega_prevloc = om_a / omega_prevloc_frac
            - estimate_times [True, False]: True exposes the tau variables
            - saclen_shift [True, False]: True makes the postsaccadic shift \
                depend of the saccade length

    """

    def __init__(self, inhib_method, att_map_init_type, shifts,
                 locdep_decay_switch, omp, data_range, kwargs_dict=None):
        # The given settings are used to pick the appropriate subfunctions of
        # the model and bind them to their general name.
        # Base Settings
        self.MAP_SIZE = 128
        self.set_precision(np.float128)

        # Internal Vars
        self.inputs_in_deg = True
        self.warn_me = False

        # Settings: picks the appropriate functions for the chosen configuration
        self.att_map_init_type = att_map_init_type  # random, cb, zero, one
        self.att_map_init_funcs = {"zero": self.initialize_map_unif,
                                   "cb" : self.initialize_center_bias}
        self.att_map_init = self.att_map_init_funcs[self.att_map_init_type]

        self.inhib_type = inhib_method  # subtractive, divisive
        self.inhib_funcs = {"subtractive": self.combine_subtractive,
                            "divisive" : self.combine_divisive}
        self.combine = self.inhib_funcs[self.inhib_type]

        self.shifts = shifts
        self.evolve_maps_funcs = {"off": self.evolve_maps_main,
                                  "pre": self.evolve_maps_presac,
                                  "post": self.evolve_maps_postsac,
                                  "both" : self.evolve_maps_both}
        self.evolve_maps = self.evolve_maps_funcs[self.shifts]

        self.locdep_decay_switch = locdep_decay_switch
        self.att_decay_funcs = {"off": self.differential_time_att_org,
                                "on" : self.differential_time_att_locdep}
        self.differential_time_att = self.att_decay_funcs[ \
            self.locdep_decay_switch]

        self.omp = omp  # "add", "mult"

        self.exponents = 2 #1, 2
        self.coupled_oms = False
        self.coupled_sigmas = False
        self.logged_cf = False
        self.logged_z = False
        self.logged_ompf = False
        self.coupled_facil = False
        self.estimate_times = False
        self.saclen_shift = False

        # Parameters
        self.omegaAttention = None
        self.omegaInhib = None
        self.sigmaAttention = None
        self.sigmaInhib = None
        self.gamma = None
        self.lamb = None
        self.inhibStrength = None
        self.zeta = None
        self.sigmaShift = None
        self.shift_size = None
        self.first_fix_OmegaAttention = None
        self.cb_sd = None #tuple (x,y)
        self.tau_pre = 50 / 1000
        self.tau_post = 100 / 1000
        self.foR_size = 2  # diameter in degrees, Not radius
        self.omega_prevloc = None
        self.chii = 0.3
        self.ompfactor = 1

        # Data
        self.data_range = data_range  #  {'x':(0, 127), 'y':(0, 127)}
        assert isinstance(self.data_range, dict)
        # add kwargs as object attributes
        if not kwargs_dict is None:
            self.__dict__.update(kwargs_dict)

    def set_mapsize(self, map_size):
        """
        Set up the operating grid size for calculation.

        Parameters
        ----------
        map_size : int
            size of the internal grid
        """
        self.MAP_SIZE = map_size
        grid = np.mgrid[0:self.MAP_SIZE, 0:self.MAP_SIZE]
        self._xx, self._yy = self.PRECISION(grid)

    def set_precision(self, data_type):
        """
        Set up the operating datatype for the whole model. np.float128 is best
        if you want to make sure the model doesnt run into numerical problems
        during estimation. Lower precision is faster though.

        Parameters
        ----------
        data_type : data-type
            datatype to be used for internal calculations
        """
        self.PRECISION = data_type
        self.EPS = np.finfo(self.PRECISION).eps
        grid = np.mgrid[0:self.MAP_SIZE, 0:self.MAP_SIZE]
        self._xx, self._yy = self.PRECISION(grid)
    # --------------------------------------------------------------------------
    # HELPER FUNCTIONS
    # --------------------------------------------------------------------------
    def check_params_in_bounds(self):
        """
        Checks if all specified parameters fall in the defined range of the
        model

        Returns
        -------
        bool
            True if all parameters are okay
        """
        basics_not_none = not (None in [self.omegaAttention, self.omegaInhib,
                                        self.sigmaAttention, self.sigmaInhib,
                                        self.gamma, self.lamb,
                                        self.inhibStrength, self.zeta])
        assert basics_not_none
        zeta_range = (0, 1) if not self.logged_z else(10 ** -20, 10 **0)
        ompf_range = (0, 10) if not self.logged_ompf else (10**-20, 10 ** 1)

        all_valid = (0 < self.omegaAttention < 10000) and \
                    (0 < self.omegaInhib < 10000) and \
                    (0 < self.sigmaAttention < 10000) and \
                    (0 < self.sigmaInhib < 10000) and \
                    (0 < self.gamma < 15) and \
                    (0 < self.lamb < 15) and \
                    (0 < self.inhibStrength < 10000) and \
                    (zeta_range[0] <= self.zeta <= zeta_range[1]) and \
                    (self.cb_sd is None or (0 < self.cb_sd[0] < 10000)) and \
                    (self.cb_sd is None or (0 < self.cb_sd[1] < 10000)) and \
                    (self.first_fix_OmegaAttention is None or \
                        (0 < self.first_fix_OmegaAttention < 10000)) and \
                    (self.tau_pre is None or (0 < self.tau_pre < 10000)) and \
                    (self.tau_post is None or (0 < self.tau_post < 10000)) and \
                    (self.sigmaShift is None or (0 < self.sigmaShift < 10000)) \
                        and \
                    (self.shift_size is None or (0 < self.shift_size < 10000)) \
                        and \
                    (self.omega_prevloc is None \
                        or (0 < self.omega_prevloc < 10000)) and \
                    (0 < self.tau_pre < 100) and \
                    (0 < self.tau_post < 100) and \
                    (ompf_range[0] < self.ompfactor < ompf_range[1]) and \
                    (0.0001 < self.chii < 2)
        return all_valid and basics_not_none

    def check_params_for_config(self):
        """
        Checks whether all necessary parameters are present for the current
        configuration.

        Returns
        -------
        bool
            True if all parameters are okay
        """
        # basic params
        assert self.omegaAttention is not None
        assert self.omegaInhib is not None
        assert self.sigmaAttention is not None
        assert self.sigmaInhib is not None
        assert self.gamma is not None
        assert self.lamb is not None
        assert self.inhibStrength is not None
        assert self.zeta is not None
        # Center bias
        if self.att_map_init_type == "cb":
            assert len(self.cb_sd) == 2
            assert self.first_fix_OmegaAttention is not None
        if self.shifts == "pre" or self.shifts == "both":
            assert self.tau_pre is not None
        if self.shifts == "post" or self.shifts == "both":
            assert self.sigmaShift is not None
            assert self.shift_size is not None
            assert self.tau_post is not None
        if self.locdep_decay_switch == "on":
            assert self.omega_prevloc is not None
        return True

    def whoami(self):
        """Returns the model identity as a string"""
        id_str = "I am a "
        id_str += self.inhib_type + " scenewalk model, "
        id_str += "initialized with " + self.att_map_init_type + " activation, "
        id_str += "in " + str(self.exponents) + " exponents mode"
        if self.shifts == "pre" or self.shifts == "both":
            id_str += ", using a presaccadic shift"
        if self.shifts == "post" or self.shifts == "both":
            id_str += ", using a postsaccadic shift"
        if self.locdep_decay_switch == "on":
            id_str += ", using location dependent attention decay"
        if self.coupled_oms:
            id_str += ", with om_i as a fraction"
        if self.coupled_sigmas:
            id_str += ", with coupled sigmas"
        if self.coupled_facil:
            id_str += ", with coupled facilitation"
        if self.logged_cf:
            id_str += ", with logged cf"
        if self.logged_z:
            id_str += ", with logged z"
        if self.saclen_shift:
            id_str += ", with eta=saclen"
        if self.omp != "off":
            id_str += ", with omp"
        if self.logged_ompf:
            id_str += "logged"
        return id_str

    def clear_params(self):
        """Clears all parameters"""
        self.omegaAttention, self.omegaInhib, self.sigmaAttention, \
        self.sigmaInhib, self.gamma, self.lamb, self.inhibStrength, self.zeta, \
        self.sigmaShift, self.shift_size, self.first_fix_OmegaAttention, \
        self.cb_sd, self.tau_pre, self.tau_post, self.foR_size, \
        self.omega_prevloc, self.chii, self.ompfactor = [None]*18

    def get_params(self):
        """Returns the current parameters as a dictionary"""
        p_list = OrderedDict({
            "omegaAttention": self.omegaAttention,
            "omegaInhib": self.omegaInhib,
            "sigmaAttention": self.sigmaAttention,
            "sigmaInhib": self.sigmaInhib,
            "gamma": self.gamma
        })
        if self.exponents == 2:
            p_list["lamb"] = self.lamb
        p_list["inhibStrength"] = self.inhibStrength
        p_list["zeta"] = self.zeta

        if self.shifts == "post" or self.shifts == "both":
            p_list["sigmaShift"] = self.sigmaShift
            p_list["shift_size"] = self.shift_size
        # Center bias
        if self.att_map_init_type == "cb":
            p_list["first_fix_OmegaAttention"] = self.first_fix_OmegaAttention
            p_list["cb_sd_x"] = self.cb_sd[0]
            p_list["cb_sd_y"] = self.cb_sd[1]
        if self.locdep_decay_switch == "on":
            p_list["omega_prevloc"] = self.omega_prevloc
        if self.estimate_times:
            p_list["tau_pre"] = self.tau_pre
            p_list["tau_post"] = self.tau_post
        if self.omp != "off":
            p_list["chi"] = self.chii
            p_list["ompfactor"] = self.ompfactor
        return p_list

    def get_param_list_order(self):
        """ Returns the names and order of the parameters required by the
        current model configuration"""
        param_names = []
        if self.coupled_oms:
            param_names.extend(["omegaAttention", "omfrac"])
        else:
            param_names.extend(["omegaAttention", "omegaInhib"])
        if self.coupled_sigmas:
            param_names.extend(["sigmaAttention"])
        else:
            param_names.extend(["sigmaAttention", "sigmaInhib"])
        param_names.extend(["gamma"])
        if self.exponents == 2:
            param_names.extend(["lamb"])
        param_names.extend(["inhibStrength", "zeta"])
        if self.shifts == "post" or self.shifts == "both":
            param_names.extend(["sigmaShift", "shift_size"])
        if self.att_map_init_type == "cb":
            param_names.extend(["first_fix_OmegaAttention", "cb_sd_x",
                                "cb_sd_y"])
        if self.locdep_decay_switch == "on":
            if self.coupled_facil:
                param_names.extend(["omega_prevloc_frac"])
            else:
                param_names.extend(["omega_prevloc"])
        if self.estimate_times:
            param_names.extend(["tau_pre"])
            param_names.extend(["tau_post"])
        if self.omp != "off":
            param_names.extend(["chi"])
            param_names.extend(["ompfactor"])
        return param_names

    def update_params(self, scene_walk_parameters):
        """
        Update model parameters

        Parameters
        ----------
        scene_walk_parameters : list or dict
            scene walk params either as list (not numpy array!) or as dict

        Returns
        -------
        left over parameters when list of the wrong size is passed.
        """
        if isinstance(scene_walk_parameters, dict):
            if self.coupled_oms:
                self.omegaAttention = scene_walk_parameters["omegaAttention"]
                omfrac = scene_walk_parameters["omfrac"]
                self.omegaInhib = self.omegaAttention / omfrac
            else:
                self.omegaAttention = scene_walk_parameters["omegaAttention"]
                self.omegaInhib = scene_walk_parameters["omegaInhib"]
            self.sigmaAttention = scene_walk_parameters["sigmaAttention"]
            if self.coupled_sigmas:
                self.sigmaInhib = scene_walk_parameters["sigmaAttention"]
            else:
                self.sigmaInhib = scene_walk_parameters["sigmaInhib"]
            self.gamma = scene_walk_parameters["gamma"]
            if self.exponents == 1:
                self.lamb = self.gamma
            else:
                self.lamb = scene_walk_parameters["lamb"]
            if self.logged_cf:
                self.inhibStrength = \
                    10 ** (scene_walk_parameters["inhibStrength"])
            else:
                self.inhibStrength = scene_walk_parameters["inhibStrength"]
            if self.logged_z:
                self.zeta = 10 ** scene_walk_parameters["zeta"]
            else:
                self.zeta = scene_walk_parameters["zeta"]
            if self.shifts == "post" or self.shifts == "both":
                self.sigmaShift = scene_walk_parameters["sigmaShift"]
                self.shift_size = scene_walk_parameters["shift_size"]
            if self.att_map_init_type == "cb":
                self.first_fix_OmegaAttention = \
                    scene_walk_parameters["first_fix_OmegaAttention"]
                self.cb_sd = \
                    (scene_walk_parameters["cb_sd_x"],
                     scene_walk_parameters["cb_sd_y"])
            if self.locdep_decay_switch == "on":
                if self.coupled_facil:
                    self.omega_prevloc = \
                        (self.omegaAttention
                         / scene_walk_parameters["omega_prevloc_frac"])
                else:
                    self.omega_prevloc = scene_walk_parameters["omega_prevloc"]
            if self.estimate_times:
                self.tau_pre = scene_walk_parameters["tau_pre"]
                self.tau_post = scene_walk_parameters["tau_post"]
            if self.omp != "off":
                self.chii = scene_walk_parameters["chi"]
                if self.logged_ompf:
                    self.ompfactor = 10 ** scene_walk_parameters["ompfactor"]
                else:
                    self.ompfactor = scene_walk_parameters["ompfactor"]
        elif isinstance(scene_walk_parameters, list):
            scene_walk_params = scene_walk_parameters.copy()
            self.omegaAttention = scene_walk_params.pop(0)
            if self.coupled_oms:
                omfrac = scene_walk_params.pop(0)
                self.omegaInhib = self.omegaAttention / omfrac
            else:
                self.omegaInhib = scene_walk_params.pop(0)
            self.sigmaAttention = scene_walk_params.pop(0)
            if self.coupled_sigmas:
                self.sigmaInhib = self.sigmaAttention
            else:
                self.sigmaInhib = scene_walk_params.pop(0)
            self.gamma = scene_walk_params.pop(0)
            if self.exponents == 1:
                self.lamb = self.gamma
            elif self.exponents == 2:
                self.lamb = scene_walk_params.pop(0)
            else:
                raise Exception("invalid exponent input")
            if self.logged_cf:
                self.inhibStrength = 10 ** (scene_walk_params.pop(0))
            else:
                self.inhibStrength = scene_walk_params.pop(0)
            if self.logged_z:
                self.zeta = 10 ** (scene_walk_params.pop(0))
            else:
                self.zeta = scene_walk_params.pop(0)
            # Extension
            if self.shifts == "post" or self.shifts == "both":
                self.sigmaShift = scene_walk_params.pop(0)
                self.shift_size = scene_walk_params.pop(0)
            if self.att_map_init_type == "cb":
                self.first_fix_OmegaAttention = scene_walk_params.pop(0)
                self.cb_sd = (scene_walk_params.pop(0),
                              scene_walk_params.pop(0))
            if self.locdep_decay_switch == "on":
                if self.coupled_facil:
                    facil_frac = scene_walk_params.pop(0)
                    self.omega_prevloc = self.omegaAttention / facil_frac
                else:
                    self.omega_prevloc = scene_walk_params.pop(0)
            if self.estimate_times:
                self.tau_pre = scene_walk_params.pop(0)
                self.tau_post = scene_walk_params.pop(0)
            if self.omp != "off":
                self.chii = scene_walk_params.pop(0)
                if self.logged_ompf:
                    self.ompfactor = 10 ** scene_walk_params.pop(0)
                else:
                    self.ompfactor = scene_walk_params.pop(0)
            if len(scene_walk_params) != 0:
                warnings.warn("You passed more parameters than your model can "
                              "use. Why would you do that?")
                return scene_walk_params
        else:
            raise Exception("Data Type Error: you need to provide the "
                            "parameters as either a dict or a list")


    def convert_deg_to_px(self, dat, dim, fix=False, cutoff=True, grid_sz=None):
        """
        Converts degree values to pixels on the grid.

        Parameters
        ----------
        dat : int or float
            number to convert in degrees
        dim : {'x' or 'y'}
            dimension along which to convert
        fix : bool
            if true, returned pixel value is between 0 and 127 (in the grid)
        cutoff : bool
            if True points outside the grid are set to the border
        grid_sz : int
            enables setting the grid size to another size than the internal \
            model size

        Returns
        -------
        float
            pixel value
        """
        if grid_sz is None:
            grid_sz = self.MAP_SIZE
        if not self.inputs_in_deg:
            return dat
        if dat is None:
            raise Exception("dat is None. Dat shouldnt be none! Are you maybe "
                            "trying to evaluate a model that has no params "
                            "yet?")
        if fix:
            dat_px = (((dat - min(self.data_range[dim])) /
                       (max(self.data_range[dim]) - min(self.data_range[dim])))
                      * grid_sz)
            dat_px = int(np.floor(dat_px))
            if cutoff:
                dat_px = grid_sz-1 if dat_px >= grid_sz else dat_px
                dat_px = 0 if dat_px <= 0 else dat_px
        else:
            dat_px = ((dat) / (max(self.data_range[dim])
                               - min(self.data_range[dim])) * grid_sz)
            dat_px = 0.1 if dat_px < 0.1 else dat_px
        return dat_px

    def convert_px_to_deg(self, dat, dim, grid_sz=None):
        """
        Converts pixel values to degrees on the grid.

        Parameters
        ----------
        dat : int or float
            number to convert in pixels
        dim : {'x' or 'y'}
            dimension along which to convert
        grid_sz : int
            enables setting the grid size to another size than the internal \
            model size

        Returns
        -------
        float
            degree value
        """
        if grid_sz is None:
            grid_sz = self.MAP_SIZE
        if not self.inputs_in_deg:
            return dat
        dat_deg = ((dat / grid_sz) * (max(self.data_range[dim])
                                      - min(self.data_range[dim]))
                   + min(self.data_range[dim]))
        return dat_deg

    def get_unit_vector(self, point1, point2):
        """
        get the unit vector between 2 points. point1 is the origin.
        Vector goes toward point 2.

        Parameters
        ----------
        point1, point2 : iterables of shape (x, y)
            points between which to find the unit vector

        Returns
        -------
        list
            [unit vector x, unit vector y]
        float
            magnitude
        """
        # delta vector
        d_x = point2[0] - point1[0]
        d_y = point2[1] - point1[1]
        # find magnitude
        vec_magnitude = np.sqrt((d_x ** 2) + (d_y ** 2))
        if vec_magnitude == 0:
            #print(point1, point2)
            if self.warn_me:
                warnings.warn("no movement between two fixations")
            u_x = 0
            u_y = 0
        else:
            # find unit vector
            u_x = d_x / vec_magnitude
            u_y = d_y / vec_magnitude
        return [u_x, u_y], vec_magnitude

    def simulate_durations(self, amount):
        """
        generate durations from gamma distribtion.

        Parameters
        ----------
        amount : int
            number of durations to output

        Returns
        -------
        array
            vector of durations
        """
        from scipy.stats import gamma
        durations = []
        for n in range(amount):
            duration = int(np.floor(gamma.rvs(a=100, scale=2.5, size=1))) / 1000
            durations.append(duration)
        return durations

    def empirical_fixation_density(self, x_locs_deg, y_locs_deg):
        """
        Calculates the empirical fixation density given fixation points and
        range information using the "scott" bandwidth (as in Schütt et al, 2017)

        Parameters
        ----------
        x_locs_deg, y_locs_deg : float
            coordinates in degrees

        Returns
        -------
        int
            x coordinates in px
        int
            y coordinates in px
        array
            fixation density map
        """
        from scipy.stats import kde
        # reduce grid to 128x128
        # watch out these are NOT indexes!!
        x_px = [self.convert_deg_to_px(x, 'x', fix=True) for x in x_locs_deg]
        y_px = [self.convert_deg_to_px(y, 'y', fix=True) for y in y_locs_deg]
        # smooth
        k = kde.gaussian_kde([x_px, y_px], bw_method="scott")
        # resolution of image grid is 128x128
        xi, yi = self.PRECISION(np.mgrid[0:self.MAP_SIZE, 0:self.MAP_SIZE])
        # apply smoothed data to grid
        zi = k(np.vstack([yi.flatten(), xi.flatten()]))
        # normalize
        zi = zi/np.sum(zi)
        zi = zi.reshape(xi.shape)
        return x_px, y_px, zi

    def fixation_picker_max(self, likelihood_map, get_lik=False):
        """
        Picks the next fixation location according to the maximum activation
        value (deterministic).

        Parameters
        ----------
        likelihood_map : array
            (usually 128x128) map from which to pick
        get_lik : bool
            if true, return the point's likelihood value

        Returns
        -------
        float
            x value in degrees
        float
            y value in degrees
        float (optional)
            likelihood
        """
        from random import choice as pick_val
        max_locs = np.argwhere(likelihood_map == np.max(likelihood_map))
        i, j = pick_val(max_locs)
        x_deg = self.convert_px_to_deg(j, 'x')
        y_deg = self.convert_px_to_deg(i, 'y')
        if get_lik:
            lik = np.log2(likelihood_map[i, j])
            return(x_deg, y_deg, lik)
        return(x_deg, y_deg)

    def fixation_picker_stoch(self, likelihood_map, get_lik=False):
        """
        Picks the next fixation location according to the cumulative probability
        method (linear selection algorithm)

        Parameters
        ----------
        likelihood_map : array
            (usually 128x128) map from which to pick
        get_lik : bool
            if true, return the point's likelihood value

        Returns
        -------
        float
            x value in degrees
        float
            y value in degrees
        float (optional)
            likelihood
        """
        likelihood_map = likelihood_map/np.sum(likelihood_map)
        r = np.random.rand()
        cu = np.cumsum(likelihood_map)
        cu = cu.reshape(likelihood_map.shape)
        i, j = np.argwhere(cu >= r)[0]
        x_deg = self.convert_px_to_deg(j, 'x')
        y_deg = self.convert_px_to_deg(i, 'y')
        if get_lik:
            lik = np.log2(likelihood_map[i, j])
            return(x_deg, y_deg, lik)
        return(x_deg, y_deg)

    # --------------------------------------------------------------------------
    # COMPONENTS
    # --------------------------------------------------------------------------
    def initialize_map_unif(self):
        """
        Initializes a map with near 0 activation everywhere

        Returns
        -------
        array
            initial starting map
        """
        map_init = self.EPS * self.PRECISION(np.ones(np.shape(self._xx)))
        return map_init

    def initialize_center_bias(self):
        """
        Initializes a map with with a central gaussian

        Returns
        -------
        array
            initial starting map
        """
        cb_sd_x = self.convert_deg_to_px(self.cb_sd[0], 'x')
        cb_sd_y = self.convert_deg_to_px(self.cb_sd[1], 'y')
        mu = np.array([64, 64])
        rad = (((self._xx - mu[0])** 2) / (cb_sd_x ** 2) +
               ((self._yy-mu[1])**2)/(cb_sd_y**2)).T
        mapAtt_init = ((1 / (2 * np.pi * cb_sd_x * cb_sd_y))
                       * np.exp(-(rad / (2 * (1 ** 2)))))
        #normalize
        mapAtt_init = mapAtt_init/np.sum(mapAtt_init)
        return mapAtt_init

    # make gauss v1
    def make_attention_gauss(self, fixs_x, fixs_y):
        """
        make gaussian window at fixation point for attention

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y
        durations : tuple
            of the shape (previous, current, next) of durations

        Returns
        -------
        array
            gaussian attention map (usually 128x128)
        """
        fix_x = fixs_x[1]
        fix_y = fixs_y[1]
        sigmaAttention_x = self.convert_deg_to_px(self.sigmaAttention, 'x')
        sigmaAttention_y = self.convert_deg_to_px(self.sigmaAttention, 'y')
        fix_x = self.convert_deg_to_px(fix_x, 'x', fix=True)
        fix_y = self.convert_deg_to_px(fix_y, 'y', fix=True)
        # equation 5
        # T fixes weird meshgrid thing
        rad = ((((self._xx - fix_x)** 2) / (sigmaAttention_x ** 2))
               + (((self._yy - fix_y)** 2) / (sigmaAttention_y ** 2))).T
        gaussAttention = ((1 / (2 * np.pi * sigmaAttention_x
                                * sigmaAttention_y))
                          * np.exp(-(rad / (2 * (1 ** 2)))))
        return gaussAttention

    # make gauss v1
    def make_attention_gauss_post_shift(self, fixs_x, fixs_y, get_loc=False):
        """
        make gaussian window at fixation point for attention during the
        post-saccadic shift

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's y
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y
        durations : tuple
            of the shape (previous, current, next) of durations

        Returns
        -------
        array
            gaussian attention map (usually 128x128)
        """
        fix_x_prev = fixs_x[0]
        fix_y_prev = fixs_y[0]
        fix_x = fixs_x[1]
        fix_y = fixs_y[1]
        # post saccadic attention shift
        sigmaShift_x = self.convert_deg_to_px(self.sigmaShift, 'x')
        sigmaShift_y = self.convert_deg_to_px(self.sigmaShift, 'y')
        # get unit vector in degrees
        u, mag = self.get_unit_vector([fix_x_prev, fix_y_prev], [fix_x, fix_y])

        if self.saclen_shift:
            shift_by = mag*self.shift_size
        else:
            shift_by = self.shift_size

        shift_loc_x = fix_x + (u[0] * shift_by)
        shift_loc_y = fix_y + (u[1] * shift_by)
        shift_loc_x_px = self.convert_deg_to_px(shift_loc_x, 'x',
                                                fix=True, cutoff=False)
        shift_loc_y_px = self.convert_deg_to_px(shift_loc_y, 'y',
                                                fix=True, cutoff=False)

        rad = (((self._xx - shift_loc_x_px)** 2) / (sigmaShift_x ** 2)
               + ((self._yy - shift_loc_y_px)** 2) / (sigmaShift_y ** 2)).T
        gaussAttention_shift = ((1 / (2 * np.pi * sigmaShift_x * sigmaShift_y))
                                * np.exp(-(rad / (2 * (1 ** 2)))))

        if np.isclose(np.sum(gaussAttention_shift), 0):
            # gaussAttention_shift = self.initialize_map_unif()
            gaussAttention_shift = self.PRECISION(np.ones(np.shape(self._xx)))
            if self.warn_me:
                warning_msg = str(["shift gauss 0. tried to put a gaussian at",
                                   shift_loc_x_px, shift_loc_y_px, "with ",
                                   sigmaShift_x, sigmaShift_y])
                warnings.warn(warning_msg)

        if get_loc:
            return gaussAttention_shift, shift_loc_x_px, shift_loc_y_px
        return gaussAttention_shift

    def combine_att_fixdens(self, gaussAttention, fix_density_map):
        """
        add empirical density information to gaussian attention mask

        Parameters
        ----------
        gaussAttention : array
            attention mask (usually 128x128)
        fix_density_map : array
            fixation desinty map (usually 128x128)

        Returns
        -------
        array
            combined attention map (usually 128x128)
        """
        assert np.sum(gaussAttention) != 0
        assert np.sum(fix_density_map) != 0
        if (fix_density_map.size != (self.MAP_SIZE, self.MAP_SIZE)) \
            and (np.sqrt(fix_density_map.size) != self.MAP_SIZE):
            raise Exception("The input fixation density map needs to be the "
                            "same size as the internal model maps set by "
                            "MAP_SIZE.")
        # equation 8
        salFixation = np.multiply(fix_density_map, gaussAttention)
        if np.sum(salFixation) == 0:
            rand_name = str(np.random.randint(1000))
            np.save((rand_name + "debugfile_gaussAtt.npy"), gaussAttention)
            np.save((rand_name + "debugfile_fix_density_map.npy"),
                    fix_density_map)
            np.save((rand_name + "debugfile_salFixation.npy"), salFixation)
        salFixation = salFixation / np.sum(salFixation)
        return salFixation

    def make_inhib_gauss(self, fixs_x, fixs_y):
        """
        make gaussian window at fixation point for inhibition

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's y
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y

        Returns
        -------
        array
            gaussian inhibition map
        """
        fix_x = fixs_x[1]
        fix_y = fixs_y[1]
        sigmaInhib_x = self.convert_deg_to_px(self.sigmaInhib, 'x')
        sigmaInhib_y = self.convert_deg_to_px(self.sigmaInhib, 'y')
        fix_x = self.convert_deg_to_px(fix_x, 'x', fix=True)
        fix_y = self.convert_deg_to_px(fix_y, 'y', fix=True)

        # equation 5
        # T fixes weird meshgrid thing
        rad = (((self._xx - fix_x)** 2) / (sigmaInhib_x ** 2)
               + ((self._yy - fix_y)** 2) / (sigmaInhib_y ** 2)).T
        gaussInhib = ((1 / (2 * np.pi * sigmaInhib_x * sigmaInhib_y))
                      * np.exp(-(rad / (2 * (1 ** 2)))))
        return gaussInhib

    def combine_subtractive(self, mapAtt, mapInhib):
        """
        Combine attention and inhibition in a subtractive way

        Parameters
        ----------
        mapAtt : array
            attention map (128x128)
        mapInhib : array
            inhibition map (128x128)

        Returns
        -------
        array
            combined map (128x128)
        """
        # equation 10
        mapAttPower = mapAtt ** self.lamb
        mapAttPowerNorm = mapAttPower / np.sum(mapAttPower)
        # handle numerical problem 1
        if np.isnan(mapAttPowerNorm).any():
            raise Exception(
                "Value of *lambda* is too large. All of mapAttPower is flushed "
                "to zero."+self.whoami()+" params "+ str(self.get_params()))

        mapInhibPower = mapInhib ** self.gamma
        mapInhibPowerNorm = mapInhibPower / np.sum(mapInhibPower)
        # handle numerical problem 1
        if np.isnan(mapInhibPowerNorm).any():
            raise Exception(
                "Value of *gamma* is too large. All of mapInhibPower is flushed"
                " to zero."+self.whoami()+" params "+ str(self.get_params()))
        u = mapAttPowerNorm - self.inhibStrength * mapInhibPowerNorm
        return u

    def combine_divisive(self, mapAtt, mapInhib):
        """
        Combine attention and inhibition in a divisive way

        Parameters
        ----------
        mapAtt : array
            attention map (128x128)
        mapInhib : array
            inhibition map (128x128)

        Returns
        -------
        array
            combined map (128x128)
        """
        mapAttPower = mapAtt ** self.lamb
        if np.sum(mapAttPower) == 0:
            raise Exception(
                "Value of *lambda* is too large. All of mapAttPower is flushed "
                "to zero."+self.whoami()+" params "+ str(self.get_params()))
        mapInhibPower = mapInhib ** self.gamma
        if np.sum(mapInhibPower) == 0:
            raise Exception(
                "Value of *lambda* is too large. All of mapAttPower is flushed"
                " to zero."+self.whoami()+" params "+ str(self.get_params()))
        # Normalizes ihib strength parameter to refer to how strong inhib is
        # compared to random activation
        inhibStrengthNorm = self.inhibStrength/(self.MAP_SIZE**2)
        # cast to float before adding to other float.
        # If we dont do this, recusion errors ensue (possibly implicit casting
        # problem?)
        inhibStrength_power = self.PRECISION((inhibStrengthNorm ** self.gamma))
        weighted_mapInhib = inhibStrength_power + mapInhibPower
        u = np.divide(mapAttPower, weighted_mapInhib)  # divisive inhibition
        return u

    def differential_time_basic(self, duration, current_map, prev_map, omega):
        """
        evolve the maps over time, given a decay paramter

        Parameters
        ----------
        duration : float
            current fixation duration in seconds
        current_map : float
            current attention or inhibition map
        prev_map : float
            previous attention or inhibition map
        omega : float
            decay parameter to use

        Returns
        -------
        array
            evolved map
        """
        current_map = current_map/np.sum(current_map)
        evolved_map = (current_map + np.exp(-duration * omega)
                       * (prev_map - current_map))
        return evolved_map

    def differential_time_att_org(self, duration, current_map, prev_map,
                                  **kwargs):
        """
        evolve the maps over time using the *attention* decay parameter

        Parameters
        ----------
        duration : float
            current fixation duration in seconds
        current_map : float
            current attention map
        prev_map : float
            previous attention map

        Returns
        -------
        array
            evolved map
        """
        current_map = current_map/np.sum(current_map)
        evolved_map = (current_map + np.exp(-duration * self.omegaAttention)
                       * (prev_map - current_map))
        return evolved_map

    def differential_time_att_locdep_round(self, duration, current_map,
                                           prev_map, fixs_x=None, fixs_y=None):
        """
        SLOWER version of differential_time_att_locdep
        evolve the maps over time with the normal attention decay parameter
        everywhere except around the previous fixation location. A circular
        aperture around the previous location defines the are of reduced decay.

        Parameters
        ----------
        duration : float
            current fixation duration in seconds
        current_map : float
            current attention map
        prev_map : float
            previous attention map
        fixs_x : tuple
            x fixation locations (prev, current, next)
        fixs_y: tuple
            y fixation location (prev, current, next)

        Returns
        -------
        array
            evolved map
        """
        assert fixs_x[0] is not None
        assert fixs_y[0] is not None
        omega_locdep = (np.ones((self.MAP_SIZE, self.MAP_SIZE))
                        * self.omegaAttention)
        j_prev = self.convert_deg_to_px(fixs_x[0], 'x', fix=True)
        i_prev = self.convert_deg_to_px(fixs_y[0], 'y', fix=True)

        r_px_x = self.convert_deg_to_px(self.foR_size/2, 'x', fix=False)
        r_px_y = self.convert_deg_to_px(self.foR_size/2, 'y', fix=False)

        mask = (((self._xx - j_prev)** 2) / (r_px_x ** 2)
                + ((self._yy - i_prev)** 2) / (r_px_y ** 2)).T < 1
        omega_locdep[mask] = self.omega_prevloc

        current_map = current_map/np.sum(current_map)
        evolved_map = (current_map + np.exp(-duration * omega_locdep)
                       * (prev_map - current_map))
        return evolved_map

    def differential_time_att_locdep(self, duration, current_map, prev_map,
                                     fixs_x=None, fixs_y=None):
        """
        QUICKER version of differential_time_att_locdep_round. Evolve the maps
        over time with the normal attention decay parameter everywhere except
        around the previous fixation location. A Rectangular aperture around the
        previous location defines the are of reduced decay.

        Parameters
        ----------
        duration : float
            current fixation duration in seconds
        current_map : float
            current attention map
        prev_map : float
            previous attention map
        fixs_x : tuple
            x fixation locations (prev, current, next)
        fixs_y: tuple
            y fixation location (prev, current, next)

        Returns
        -------
        array
            evolved map
        """
        assert fixs_x[0] is not None
        assert fixs_y[0] is not None
        omega_locdep = (np.ones((self.MAP_SIZE, self.MAP_SIZE))
                        * self.omegaAttention)
        r_px_x = int(np.floor(self.convert_deg_to_px(self.foR_size / 2, 'x',
                                                     fix=False)))
        r_px_y = int(np.floor(self.convert_deg_to_px(self.foR_size / 2, 'y',
                                                     fix=False)))
        j_prev = self.convert_deg_to_px(fixs_x[0], 'x', fix=True)
        i_prev = self.convert_deg_to_px(fixs_y[0], 'y', fix=True)

        omega_locdep[np.clip(i_prev - r_px_y, 0, None) : i_prev + r_px_y + 1,
                     np.clip(j_prev - r_px_x, 0, None): j_prev + r_px_x + 1] = \
                         self.omega_prevloc

        current_map = current_map/np.sum(current_map)
        evolved_map = (current_map + np.exp(-duration * omega_locdep)
                       * (prev_map - current_map))
        return evolved_map


    def make_positive(self, u):
        """
        cuts off negative components

        Parameters
        ----------
        u : array
            map to cut

        Returns
        -------
        array :
            ustar
        """
        # equation 12
        ustar = u
        ustar[ustar <= 0] = 0
        # handle numerical problem 3:
        # if inhibStrength is too large, whole map can be negative,
        # whole map is set to 0, cannot divide by sum 0
        # solution: make uniform
        if np.sum(ustar) == 0:
            ustar[:] = 1
        ustar = ustar / np.sum(ustar)
        return ustar

    def add_noise(self, ustar):
        """
        adds zeta noise

        Parameters
        ----------
            ustar : array
                map without noise

        Returns
        -------
        array
            uFinal; map with noise
        """
        #ustar=ustar**2
        uFinal = ((1 - self.zeta) * ustar + self.zeta
                  / np.prod(np.shape(self._xx)))
        #uFinal = uFinal**2
        return uFinal

    def get_phase_times_both(self, nth, durations):
        """
        Helper function that computes the durations of each phase
        1. You always have the post phase, no matter what
        2. the main phase can be skipped if the post phase is already too long
        3. from the time that is left for the main phase we then subtract the \
            pre phase. If that leaves the main phase less than 10 ms short, \
            we skip the pre phase.
        4. The post phase is always skipped for the first fixation

        Parameters
        ----------
        nth : int
            the how many'th fixation in the sequence are we dealing with
        durations : tuple
            of the shape (previous, current, next) of durations

        Returns
        -------
        float
            duration of post phase (secs)
        float
            duration of main phase (secs)
        float
            duration of pre phase (secs)
        """
        if nth > 1:
            if durations[1] > self.tau_post:
                duration_post_ph = self.tau_post
                if (durations[1] - duration_post_ph) > \
                    (self.tau_pre + (10 / 1000)):
                    duration_pre_ph = self.tau_pre
                else:
                    duration_pre_ph = 0
                duration_main_ph = (durations[1] - duration_pre_ph
                                    - duration_post_ph)
            else:
                duration_post_ph = durations[1]
                duration_main_ph = 0
                duration_pre_ph = 0
        else:
            duration_post_ph = 0
            if durations[1] > self.tau_pre + (10 / 1000):
                duration_pre_ph = self.tau_pre
                duration_main_ph = durations[1] - duration_pre_ph
            else:
                duration_pre_ph = 0
                duration_main_ph = durations[1]
        return duration_post_ph, duration_main_ph, duration_pre_ph

    def make_om_potential(self, fix_x, fix_y):
        """
        makes an occulomotor potential map where the cardinal directions have
        higher activation than the oblique.

        Parameters
        ----------
        fix_x : float
            x coordinate on which to center the cross
        fix_y : float
            y coordinate on which to center the cross

        Returns
        -------
        array
            map with oculomotor potential
        """
        x_px = self.convert_deg_to_px(fix_x, 'x', fix=True)
        y_px = self.convert_deg_to_px(fix_y, 'y', fix=True)
        #chii = 0.08
        q1 = ((self._xx-x_px))**2
        q2 = ((self._yy-y_px))**2
        q = ((q1 * q2)**self.chii).T
        q = q / np.max(q)
        q = np.abs(q-1)
        #q = q / np.sum(q)
        return q

    def make_om_potential_neg(self, fix_x, fix_y):
        """
        makes an occulomotor potential map where the cardinal directions have
        lower activation than the oblique.

        Parameters
        ----------
        fix_x : float
            x coordinate on which to center the cross
        fix_y : float
            y coordinate on which to center the cross

        Returns
        -------
        array
            map with oculomotor potential
        """
        x_px = self.convert_deg_to_px(fix_x, 'x', fix=True)
        y_px = self.convert_deg_to_px(fix_y, 'y', fix=True)
        #chii = 0.3
        q1 = ((self._xx-x_px))**2
        q2 = ((self._yy-y_px))**2
        q = ((q1 * q2)**self.chii).T
        q = q / np.max(q)
        return q

    # --------------------------------------------------------------------------
    # MECHANISMS
    # --------------------------------------------------------------------------
    def evolve_maps_main(self, durations, fixs_x, fixs_y, map_att_prev,
                         map_inhib_prev, fix_density_map, nth, sim=False):
        """
        Evolve maps with basic model

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x in
            degrees
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y in
            degrees
        durations : tuple
            of the shape (previous, current, next) of durations in seconds
        map_att_prev : array
            previous attention map
        map_inhib_prev : array
            previous inhibition map
        fix_density_map : array
            empirical fixation density of the image
        nth : int
            number of the fixation in the sequence

        Returns
        -------
        array
            attention map
        array
            inhibition map
        array
            final map
        tuple
            coordinates for the next fixation in degrees (x, y)
        float
            likelihood of the next fixation
        """
        # no locdep decay on first fix
        map_att = self.make_attention_gauss(fixs_x, fixs_y)
        map_att = self.combine_att_fixdens(map_att, fix_density_map)
        # on the first fixation use center bias decay or regular
        if nth == 1:
            if self.att_map_init_type == "cb":
                assert not np.isnan(self.first_fix_OmegaAttention)
                map_att = \
                    self.differential_time_basic(durations[1], map_att,
                                                 map_att_prev,
                                                 self.first_fix_OmegaAttention)
            else:
                map_att = self.differential_time_basic(durations[1], map_att,
                                                       map_att_prev,
                                                       self.omegaAttention)
        else:
            map_att = self.differential_time_att(durations[1], map_att,
                                                 map_att_prev, fixs_x=fixs_x,
                                                 fixs_y=fixs_y)

        map_inhib = self.make_inhib_gauss(fixs_x, fixs_y)
        map_inhib = self.differential_time_basic(durations[1], map_inhib,
                                                 map_inhib_prev,
                                                 self.omegaInhib)
        u = self.combine(map_att, map_inhib)

        if self.omp == "add":
            #print("adding omp")
            #u = u * (1 * self.make_om_potential(fixs_x[1], fixs_y[1]))
            #ompfactor = 2
            omp_map = self.make_om_potential(fixs_x[1], fixs_y[1])
            omp_map = omp_map/np.sum(omp_map)
            # additive OMP
            u = u + (self.ompfactor * omp_map)
        if self.omp == "mult":
            #ompfactor = 2
            # multiplicative OMP
            u = u * (self.ompfactor * self.make_om_potential(fixs_x[1],
                                                             fixs_y[1]))

        ustar = self.make_positive(u)
        uFinal = self.add_noise(ustar)
        # get likelihood for next fixations
        if (fixs_x[2] is None) & (sim):
            x, y, LL = self.fixation_picker_stoch(uFinal, get_lik=True)
            next_fix = (x, y)
        else:
            idx_j_next = self.convert_deg_to_px(fixs_x[2], "x", fix=True)
            idx_i_next = self.convert_deg_to_px(fixs_y[2], "y", fix=True)
            LL = np.log2(uFinal[idx_i_next, idx_j_next])
            next_fix = (fixs_x[2], fixs_y[2])
        return map_att, map_inhib, uFinal, next_fix, LL

    def evolve_maps_postsac(self, durations, fixs_x, fixs_y, map_att_prev,
                            map_inhib_prev, fix_density_map, nth, sim=False):
        """
        Evolve maps with postsaccadic shift.

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x in
            degrees
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y in
            degrees
        durations : tuple
            of the shape (previous, current, next) of durations in seconds
        map_att_prev : array
            previous attention map
        map_inhib_prev : array
            previous inhibition map
        fix_density_map : array
            empirical fixation density of the image
        nth : int
            number of the fixation in the sequence

        Returns
        -------
        array
            attention map
        array
            inhibition map
        array
            final map
        tuple
            coordinates for the next fixation in degrees (x, y)
        float
            likelihood of the next fixation
        """
        # During Post phase we always have locdep decay. Post phase is always
        # skipped on nth=1 anyway.
        skip_post = False
        # skip_main = False

        # If the duration of the fixation is shorter than the discrete post
        # shift time, just skip the main phase
        if durations[1] > self.tau_post:
            duration_post_ph = self.tau_post
            duration_main_ph = durations[1]-self.tau_post
        else:
            duration_post_ph = durations[1]
            duration_main_ph = 0
            # skip_main = True
        # on the first fixation we always skip the post phase (because no
        # previous fixation exists)
        if nth == 1:
            skip_post = True
            duration_main_ph = durations[1]

        # POST PHASE
        if not skip_post:
            map_att_post_shift = self.make_attention_gauss_post_shift(fixs_x,
                                                                      fixs_y)
            map_att_post_shift = self.combine_att_fixdens(map_att_post_shift,
                                                          fix_density_map)
            map_att_post_shift = self.differential_time_att(duration_post_ph,
                                                            map_att_post_shift,
                                                            map_att_prev,
                                                            fixs_x=fixs_x,
                                                            fixs_y=fixs_y)
            map_inhib_post_shift = self.make_inhib_gauss(fixs_x, fixs_y)
            map_inhib_post_shift = \
                self.differential_time_basic(duration_post_ph,
                                             map_inhib_post_shift,
                                             map_inhib_prev, self.omegaInhib)
        else:
            map_att_post_shift = map_att_prev
            map_inhib_post_shift = map_inhib_prev

        # MAIN PHASE
        # could use a skip main if statement, but then we need to combine maps
        # in this function
        durations_dummy = (None, duration_main_ph, None)
        map_att, map_inhib, uFinal, next_fix, LL = \
            self.evolve_maps_main(durations_dummy, fixs_x, fixs_y,
                                  map_att_post_shift, map_inhib_post_shift,
                                  fix_density_map, nth, sim=sim)

        return map_att, map_inhib, uFinal, next_fix, LL

    def evolve_maps_presac(self, durations, fixs_x, fixs_y, map_att_prev,
                           map_inhib_prev, fix_density_map, nth, sim=False):
        """
        Evolve maps with presaccadic shift.

        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x in
            degrees
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y in
            degrees
        durations : tuple
            of the shape (previous, current, next) of durations in seconds
        map_att_prev : array
            previous attention map
        map_inhib_prev : array
            previous inhibition map
        fix_density_map : array
            empirical fixation density of the image
        nth : int
            number of the fixation in the sequence

        Returns
        -------
        array
            attention map
        array
            inhibition map
        array
            final map
        tuple
            coordinates for the next fixation in degrees (x, y)
        float
            likelihood of the next fixation
        """
        skip_pre = False
        if durations[1] > self.tau_pre + (10/1000):
            duration_main_ph = durations[1] - self.tau_pre
            duration_pre_ph = self.tau_pre
        else:
            duration_main_ph = durations[1]
            skip_pre = True

        # MAIN PHASE
        durations_dummy = (None, duration_main_ph, None)
        map_att_main, map_inhib_main, uFinal_main, next_fix, LL = \
            self.evolve_maps_main(durations_dummy, fixs_x, fixs_y, map_att_prev,
                                  map_inhib_prev, fix_density_map, nth, sim=sim)
        # pick the next location
        if sim:
            fixs_x = (*fixs_x[0:2], next_fix[0])
            fixs_y = (*fixs_y[0:2], next_fix[1])

        # PRE PHASE
        if not skip_pre:
            # get gauss centered around the upcoming location
            map_att_shift = self.make_attention_gauss(fixs_x[1:3], fixs_y[1:3])
            map_att_shift = self.combine_att_fixdens(map_att_shift,
                                                     fix_density_map)
            # no location dependent decay on fixation 1
            if not nth == 1:
                map_att_pre = self.differential_time_att(duration_pre_ph,
                                                         map_att_shift,
                                                         map_att_main,
                                                         fixs_x=fixs_x,
                                                         fixs_y=fixs_y)
            else:
                map_att_pre = \
                    self.differential_time_basic(duration_pre_ph, map_att_shift,
                                                 map_att_main,
                                                 self.omegaAttention)

            map_inhib_pre = self.make_inhib_gauss(fixs_x, fixs_y)
            map_inhib_pre = self.differential_time_basic(duration_pre_ph,
                                                         map_inhib_pre,
                                                         map_inhib_main,
                                                         self.omegaInhib)
            u = self.combine(map_att_pre, map_inhib_pre)
            ustar = self.make_positive(u)
            uFinal_pre = self.add_noise(ustar)
        else:
            uFinal_pre = uFinal_main
            map_inhib_pre = map_inhib_main
            map_att_pre = map_att_main

        # we return the state of after the presaccadic shift but the LL and
        # picked fixation from before
        return map_att_pre, map_inhib_pre, uFinal_pre, next_fix, LL

    def evolve_maps_both(self, durations, fixs_x, fixs_y, map_att_prev,
                         map_inhib_prev, fix_density_map, nth, sim=False):
        """
        Evolve maps with pre- and postsaccadic shift.
        Parameters
        ----------
        fixs_x : tuple
            of the shape (previous, current, next) fixation location's x in
            degrees
        fixs_y : tuple
            of the shape (previous, current, next) fixation location's y in
            degrees
        durations : tuple
            of the shape (previous, current, next) of durations in seconds
        map_att_prev : array
            previous attention map
        map_inhib_prev : array
            previous inhibition map
        fix_density_map : array
            empirical fixation density of the image
        nth : int
            number of the fixation in the sequence

        Returns
        -------
        array
            attention map
        array
            inhibition map
        array
            final map
        tuple
            coordinates for the next fixation in degrees (x, y)
        float
            likelihood of the next fixation
        """
        duration_post_ph, duration_main_ph, duration_pre_ph = \
            self.get_phase_times_both(nth, durations)
        # both mechanisms
        skip_post = True if duration_post_ph == 0 else False
        skip_pre = True if duration_pre_ph == 0 else False

        # POST PHASE
        if not skip_post:
            map_att = self.make_attention_gauss_post_shift(fixs_x, fixs_y)
            map_att = self.combine_att_fixdens(map_att, fix_density_map)
            map_att = self.differential_time_att(duration_post_ph, map_att,
                                                 map_att_prev, fixs_x=fixs_x,
                                                 fixs_y=fixs_y)
            map_att_prev = map_att

            map_inhib = self.make_inhib_gauss(fixs_x, fixs_y)
            map_inhib = self.differential_time_basic(duration_post_ph,
                                                     map_inhib, map_inhib_prev,
                                                     self.omegaInhib)
            map_inhib_prev = map_inhib
        # MAIN PHASE
        durations_dummy = (None, duration_main_ph, None)
        map_att_main, map_inhib_main, uFinal_main, next_fix, LL = \
            self.evolve_maps_main(durations_dummy, fixs_x, fixs_y, map_att_prev,
                                  map_inhib_prev, fix_density_map, nth, sim=sim)

        # pick the next location
        if sim:
            fixs_x = (*fixs_x[0:2], next_fix[0])
            fixs_y = (*fixs_y[0:2], next_fix[1])

        # PRE PHASE
        if not skip_pre:
            # get gauss centered around the upcoming location
            map_att_shift = self.make_attention_gauss(fixs_x[1:3], fixs_y[1:3])
            map_att_shift = self.combine_att_fixdens(map_att_shift,
                                                     fix_density_map)
            # no locdep decay on fix nr 1
            if nth != 1:
                map_att_pre = self.differential_time_att(duration_pre_ph,
                                                         map_att_shift,
                                                         map_att_main,
                                                         fixs_x=fixs_x,
                                                         fixs_y=fixs_y)
            else:
                map_att_pre = self.differential_time_basic(duration_pre_ph,
                                                           map_att_shift,
                                                           map_att_main,
                                                           self.omegaAttention)
            map_inhib_pre = self.make_inhib_gauss(fixs_x, fixs_y)
            map_inhib_pre = self.differential_time_basic(duration_pre_ph,
                                                         map_inhib_pre,
                                                         map_inhib_main,
                                                         self.omegaInhib)
            u = self.combine(map_att_pre, map_inhib_pre)
            ustar = self.make_positive(u)
            uFinal_pre = self.add_noise(ustar)
        else:
            uFinal_pre = uFinal_main
            map_inhib_pre = map_inhib_main
            map_att_pre = map_att_main

        # we return the state of after the presaccadic shift but the LL and
        # picked fixation from before
        return map_att_pre, map_inhib_pre, uFinal_pre, next_fix, LL

    # --------------------------------------------------------------------------
    # Main Interface Functions
    # --------------------------------------------------------------------------
    def get_scanpath_likelihood(self, x_path, y_path, dur_path, fix_dens):
        """
        calculate likelihood of one scanpath under scenewalk model with params
        in scene_walk_params.

        Parameters
        ----------
        x_path : array
            array with a datapoint for each fixation's x coordinate
        y_path : array
            array with a datapoint for each fixation's y coordinate
        dur_path : array
            array with a datapoint for each fixation's duration
        fix_dens : array
            empirical fixation density of the viewed image

        Returns
        -------
        float
            sum log likelihood of the scanpath

        Notes
        -----
        Beware: it HAS to be the sum likelihood. Even if all the scan paths have
        different lengths. Don't be tempted to use the average, it messes up the
        parameter estimation procedure!
        """
        log_ll = []
        # initializations
        mapAtt = self.att_map_init()
        mapInhib = self.initialize_map_unif()
        # iterate over all fixations
        x_iter = self.window(x_path)
        y_iter = self.window(y_path)
        dur_iter = self.window(dur_path)
        i_fix = 1
        # we're always evaluating the next fixation, so the last fixation's
        # map is useless
        for fixs_x, fixs_y, durs in list(zip(x_iter, y_iter, dur_iter))[0:-1]:
            # evolve map given fixation
            mapAtt, mapInhib, _, _, LL = self.evolve_maps(durs, fixs_x, fixs_y,
                                                          mapAtt, mapInhib,
                                                          fix_dens, i_fix)
            log_ll.append(LL)
            i_fix += 1
        if np.isnan(log_ll).any():
            raise Exception("LLs are Nan :( " + self.whoami() + " params "
                            + str(self.get_params()))
        sum_log_ll = np.sum(log_ll)
        return sum_log_ll

    def simulate_scanpath(self, dur_path, fix_dens, startpos, get_LL=False):
        """
        Simulates a scanpath given durations.

        Parameters
        ----------
        dur_path : array
            array with a datapoint for each fixation's duration
        fix_dens : array
            empirical fixation density of the viewed image
        startpos : tuple
            beginning location for the scanpath in degrees (x, y)
        get_LL : bool
            whether to return the LL of the path

        Returns
        -------
        array
            x of the simulated scanpath
        array
            y of the simulated scanpath
        array (optional)
            likelihood of the simulated scanpath
        """
        ll_path = []
        x_path = []
        y_path = []
        # initializations
        mapAtt = self.att_map_init()
        mapInhib = self.initialize_map_unif()

        i_fix = 1
        prev_x = None
        prev_y = None
        curr_x = startpos[0]
        curr_y = startpos[1]
        # we're always evaluating the next fixation, so the last fixation's map
        # is useless
        for durs in self.window(dur_path):
            fixs_x = (prev_x, curr_x, None)
            fixs_y = (prev_y, curr_y, None)
            # evolve map given fixation
            mapAtt, mapInhib, _, next_fix, LL = self.evolve_maps(durs, fixs_x,
                                                                 fixs_y, mapAtt,
                                                                 mapInhib,
                                                                 fix_dens,
                                                                 i_fix,
                                                                 sim=True)
            i_fix += 1
            x_path.append(curr_x)
            y_path.append(curr_y)
            prev_x, prev_y = curr_x, curr_y
            curr_x, curr_y = next_fix
            ll_path.append(LL)
        if get_LL:
            return np.asarray(x_path), np.asarray(y_path), np.mean(ll_path)
        else:
            return np.asarray(x_path), np.asarray(y_path)

    def window(self, iterable, n_before=1, n_after=1):
        """
        Sliding window iterator. Returns each element and it's neighbors.
        None elelments returned as neighbors at beginning and end.

        Parameters
        ----------
        iterable : list-ish
            list or other kind of iterable
        n_before : int
            number of elements to return before each element
        n_after : int
            number of elements to return after each element

        Returns
        -------
        iterator
            that returns each value in the list and it's surrounding n
        """
        size = n_before + n_after + 1
        none_vals_before = [None] * n_before
        none_vals_after = [None] * n_after
        i = iter([*none_vals_before, *iterable, *none_vals_after])
        win = []
        for e in range(0, size):
            win.append(next(i))
        yield win
        for e in i:
            win = win[1:] + [e]
            yield win
