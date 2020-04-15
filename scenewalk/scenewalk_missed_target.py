"""
SceneWalk variant that has no remapping but instead always makes saccades that are too short. Corrective saccades may explain the Fish.
"""
from scenewalk.scenewalk_model_object import scenewalk as sw_obj
import numpy as np

class missed_target_sw(sw_obj):
    def __init__(self, data_range):
        super().__init__("subtractive", "zero", "both", "on", "off", data_range, {"exponents" : 1, "coupled_oms":True, "coupled_facil":True})
        self.evolve_maps = self.evolve_maps_presac_missed

    def whoami(self):
        return "CORRSAC: " + super().whoami()

    def make_attention_gauss_pre(self, fixs_x, fixs_y):
        """
        make gaussian window at fixation point for attention
        Inputs
            - fixs_y, fixs_y, durations: tuples of the shape (previous, current, next) fixation locations in degrees
        Outputs
            - gaussian attention map [128x128]
        """
        fix_x = fixs_x[1]
        fix_y = fixs_y[1]
        sigmaAttention_x = self.convert_deg_to_px(self.sigmaShift, 'x')
        sigmaAttention_y = self.convert_deg_to_px(self.sigmaShift, 'y')
        fix_x = self.convert_deg_to_px(fix_x, 'x', fix=True)
        fix_y = self.convert_deg_to_px(fix_y, 'y', fix=True)
        # equation 5
        # T fixes weird meshgrid thing
        rad = ((((self._xx - fix_x)** 2) / (sigmaAttention_x ** 2)) + (((self._yy - fix_y)** 2) / (sigmaAttention_y ** 2))).T
        gaussAttention = (1 / (2 * np.pi * sigmaAttention_x * sigmaAttention_y)) * np.exp(-(rad / (2 * (1 ** 2))))
        return gaussAttention

    def evolve_maps_presac_missed(self, durations, fixs_x, fixs_y, map_att_prev, map_inhib_prev, fix_density_map, nth, sim=False):
        #print("ypir")
        skip_pre = False
        if durations[1] > self.tau_pre + (10/1000):
            duration_main_ph = durations[1] - self.tau_pre
            duration_pre_ph = self.tau_pre
        else:
            duration_main_ph = durations[1]
            skip_pre = True

        # MAIN PHASE
        durations_dummy = (None, duration_main_ph, None)
        map_att_main, map_inhib_main, uFinal_main, next_fix, LL = self.evolve_maps_main(durations_dummy, fixs_x, fixs_y, map_att_prev, map_inhib_prev, fix_density_map, nth, sim=sim)

        # pick the next location
        if sim:
            # the location we just got is the one we're aiming for
            u, mag = self.get_unit_vector([fixs_x[1], fixs_y[1]], next_fix)
            shift_by = mag * self.shift_size
            u_reverse = [-u[0], -u[1]]
            aimed_fix_x = next_fix[0]
            aimed_fix_y = next_fix[1]
            landed_fix_x = next_fix[0]+(u_reverse[0]*shift_by)
            landed_fix_y = next_fix[1]+(u_reverse[1]*shift_by)
            landed_fix = [landed_fix_x, landed_fix_y]
            # we actually need the LL of the fixation we ended up in (not the one we aimed for)
            landed_fix_j = self.convert_deg_to_px(landed_fix_x, 'x', fix=True)
            landed_fix_i = self.convert_deg_to_px(landed_fix_y, 'y', fix=True)
            LL = np.log2(uFinal_main[landed_fix_i, landed_fix_j])
        else:
            landed_fix = next_fix
            u, mag = self.get_unit_vector([fixs_x[1], fixs_y[1]], next_fix)
            shift_by = mag * self.shift_size
            aimed_fix_x = landed_fix[0]+(u[0]*shift_by)
            aimed_fix_y = landed_fix[1]+(u[1]*shift_by)
        #aimed_fix = [aimed_fix_x, aimed_fix_y]
        
        ### Here we need a step that will determine the slightly out from the next loc
        
        # PRE PHASE
        if not skip_pre:
            # get gauss centered around aimed for location
            # we give it the current to next fixes (we want the location shifted for the vector current (1)-> next (2))
            # its going to make you a gaussian out from the next, which is what we are "aiming for" with the saccade
            map_att_shift = self.make_attention_gauss_pre([None, aimed_fix_x, None], [None, aimed_fix_y, None])
            map_att_shift = self.combine_att_fixdens(map_att_shift, fix_density_map)
            # no location dependent decay on fixation 1
            if not nth == 1:
                map_att_pre = self.differential_time_att(duration_pre_ph, map_att_shift, map_att_main, fixs_x=fixs_x, fixs_y=fixs_y)
            else:
                map_att_pre = self.differential_time_basic(duration_pre_ph, map_att_shift, map_att_main, self.omegaAttention)

            map_inhib_pre = self.make_inhib_gauss(fixs_x, fixs_y)
            map_inhib_pre = self.differential_time_basic(duration_pre_ph, map_inhib_pre, map_inhib_main, self.omegaInhib)
            u = self.combine(map_att_pre, map_inhib_pre)
            ustar = self.make_positive(u)
            uFinal_pre = self.add_noise(ustar)
        else:
            uFinal_pre = uFinal_main
            map_inhib_pre = map_inhib_main
            map_att_pre = map_att_main

        # we return the state of after the presaccadic shift but the LL and picked fixation from before
        return map_att_pre, map_inhib_pre, uFinal_pre, landed_fix, LL