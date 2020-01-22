"""
SceneWalk variant that has access to limited history at each fixation
"""
from scenewalk.scenewalk_model_object import scenewalk as sw
import numpy as np
class limited_sw(sw):
    """
    SceneWalk variant that has access to limited history at each fixation
    """
    def __init__(self, data_range, n_history):
        super().__init__("subtractive", "zero", "off", 1, "off", data_range, {"coupled_oms": True})
        self.n_history = n_history

    def whoami(self):
        return "Limited History SW: " + super().whoami()

    def get_scanpath_likelihood(self, x_path, y_path, dur_path, fix_dens):
        """
        calculate likelihood of one scanpath under scenewalk model with params in scene_walk_params.
        Inputs:
            - x_path, y_path, dur_path: vectors of the same length, with a datapoint for each fixation for x and y coordinates and duration
            - fix_dens: empirical fixation density of the viewed image
        Returns:
            - average log likelihood of the scanpath (avg because scanpaths are different lengths)
        """
        log_ll = []
        # initializations
        mapAtt = self.att_map_init()
        mapInhib = self.initialize_map_unif()

        # iterate over all fixations
        x_iter, y_iter, dur_iter = self.window(x_path), self.window(y_path), self.window(dur_path)
        i_fix = 1
        x_path2 = x_path.copy()
        y_path2 = y_path.copy()
        dur_path2 = dur_path.copy()

        for fixs_x, fixs_y, durs in list(zip(x_iter, y_iter, dur_iter))[0:-1]:
            # This is the regular loop that occurs normally
            if i_fix <= self.n_history:
                print("we want the LL of ifix ", str(i_fix+1))
                print("\t old + eval triplet:", str(fixs_x))
                # evolve map given fixation
                mapAtt, mapInhib, _, _, LL = self.evolve_maps(durs, fixs_x, fixs_y, mapAtt, mapInhib, fix_dens, i_fix)
                log_ll.append(LL)
            # in this loop we restart the evolution from the point of n history to erase any trace of previous fixations
            # we bootstrap the start of the model by cutting the path and reevaluating
            else:
                print("we want the LL of ifix"+str(i_fix+1))
                # move the front of the path by one each time
                x_path2 = x_path2[1:]
                y_path2 = y_path2[1:]
                dur_path2 = dur_path2[1:]
                # set the anding point
                x_path3 = x_path2[:self.n_history + 1]
                y_path3 = y_path2[:self.n_history + 1]
                dur_path3 = dur_path2[:self.n_history + 1]
                # initializations of bootstrap maps and paths
                mapAtt2 = self.att_map_init()
                mapInhib2 = self.initialize_map_unif()
                x_iter2, y_iter2, dur_iter2 = self.window(x_path3), self.window(y_path3), self.window(dur_path3)#
                i_fix2 = 1
                for fixs_x2, fixs_y2, durs2 in list(zip(x_iter2, y_iter2, dur_iter2))[0:-1]:
                    print("\teval triplet:", fixs_x2)
                    # calculate the maps but ignore LL until the last round
                    mapAtt2, mapInhib2, _, _, LL2 = self.evolve_maps(durs2, fixs_x2, fixs_y2, mapAtt2, mapInhib2, fix_dens, i_fix2)
                    i_fix2 += 1
                # append only the last LL
                log_ll.append(LL2)
                # clean up, just in case
                mapAtt2 = None
                mapInhib2 = None

            i_fix += 1
        # average Log Likelihood: needs to be average and not sum because scanpaths have different lengths
        if np.isnan(log_ll).any():
            raise Exception("LLs are Nan :( "+self.whoami()+" params "+ str(self.get_params()))
        sum_log_ll = np.sum(log_ll)
        return sum_log_ll
