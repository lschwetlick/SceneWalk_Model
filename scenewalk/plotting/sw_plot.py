"""
Visualizing Scenewalk

Lisa Schwetlick 2019

University of Potsdam
"""
import matplotlib.pyplot as plt
from scenewalk import scenewalk_model_object as scenewalk
import numpy as np

def plot_3_maps(ufinal_map=None, inhib_map=None, att_map=None):
    """
    Makes a nice plot of the three output maps (attention, inhibition, and
    final)

    Parameters
    ----------
    ufinal_map : array
        final SceneWalk map
    inhib_map : array
        SceneWalk inhibition map
    att_map : array
        SceneWalk attention map

    Notes
    -----
    For example see demo/detailed_look_at_sw.ipynb
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title("Final")
    ax2.set_title("Inhibition")
    ax3.set_title("Attention")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    # plot final map
    if ufinal_map is not None:
        ax1.imshow(np.float64(ufinal_map), origin="lower")
    # plot inhibition map
    if inhib_map is not None:
        ax2.imshow(np.float64(inhib_map), origin="lower")
    # plot attention map
    if att_map is not None:
        ax3.imshow(np.float64(att_map), origin="lower")
    plt.show()



def plot_dynamic_shifts(sw, fix_density_map, x_path, y_path, dur_path,
                        filename):
    """
    Make dynamic video plot of model evolution given a scanpath when the
    presaccadic attention shift is switched on.

    Parameters
    ----------
    sw : scenewalk model object
        scenewalk model object
    fix_density_map : array
        empirical density
    x_path, y_path, dur_path : arrays
        list of datapoints, one for each fixation for x and y coordinates
        and duration
    filename : str
        where to save. must end in ".mp4"

    Saves Result
    """
    import matplotlib.animation as animation

    if sw.shifts == "both":
        basic = False
    elif sw.shifts == "off":
        basic = True
    else:
        raise Exception("This model variant is not implemented")

    # initializations
    map_att_prev = sw.att_map_init()
    map_inhib_prev = sw.initialize_map_unif()

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []

    # set up figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
    ax1.set_title("Fixation Selection")
    ax2.set_title("Inhibition")
    ax3.set_title("Attention")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')

    ax1.set_xlim((0, 127))
    ax1.set_ylim((0, 127))
    ax2.set_xlim((0, 127))
    ax2.set_ylim((0, 127))
    ax3.set_xlim((0, 127))
    ax3.set_ylim((0, 127))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    abs_fix_time = 0
    abs_time = 0
    x_path_px = \
        np.asarray([sw.convert_deg_to_px(x, 'x', fix=True) for x in x_path])
    y_path_px = \
        np.asarray([sw.convert_deg_to_px(y, 'y', fix=True) for y in y_path])

    x_iter = sw.window(x_path)
    y_iter = sw.window(y_path)
    dur_iter = sw.window(dur_path)
    i_fix = 1
    # we're always evaluating the next fixation, so the last fixation's map is
    # useless
    for fixs_x, fixs_y, durs in list(zip(x_iter, y_iter, dur_iter))[0:-1]:
        print(i_fix)
        if basic:
            duration_post_ph, duration_main_ph, duration_pre_ph = \
                (0, durs[1], 0)
        else:
            duration_post_ph, duration_main_ph, duration_pre_ph = \
                sw.get_phase_times_both(i_fix, durs)
        # both mechanisms
        skip_post = True if duration_post_ph == 0 else False
        skip_pre = True if duration_pre_ph == 0 else False

        # POST PHASE
        if not skip_post:
            for ms in np.arange(0, duration_post_ph, 10 / 1000):
                map_att, shift_loc_x_px, shift_loc_y_px = \
                    sw.make_attention_gauss_post_shift(fixs_x,
                                                       fixs_y,
                                                       get_loc=True)
                map_att = sw.combine_att_fixdens(map_att, fix_density_map)
                map_att = sw.differential_time_att(ms, map_att, map_att_prev,
                                                   fixs_x=fixs_x, fixs_y=fixs_y)
                map_att_prev = map_att

                map_inhib = sw.make_inhib_gauss(fixs_x, fixs_y)
                map_inhib = sw.differential_time_basic(ms, map_inhib,
                                                       map_inhib_prev,
                                                       sw.omegaInhib)
                map_inhib_prev = map_inhib

                u = sw.combine(map_att, map_inhib)
                ustar = sw.make_positive(u)
                uFinal = sw.add_noise(ustar)
                # plot final map
                im1 = ax1.imshow(np.float64(uFinal), animated=True,
                                 origin="lower")  # , vmin=0.000061
                im2, = ax1.plot(x_path_px[range(0, i_fix)],
                                y_path_px[range(0, i_fix)],
                                c="r", animated=True)
                im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                                  y_path_px[range(0, i_fix)],
                                  c="r", s=3, animated=True)
                # plot inhibition map
                im4 = ax2.imshow(np.float64(map_inhib),
                                 animated=True, origin="lower", cmap="YlGnBu")
                # plot attention map
                im7 = ax3.imshow(np.float64(map_att),
                                 animated=True, origin="lower", cmap="YlOrRd")
                # Denote center of gaussians
                im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1],
                                          c="black", s=4, animated=True)
                im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black", s=4,
                                          animated=True)
                im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black", s=4,
                                          animated=True)

                im_rm1 = ax1.scatter(shift_loc_x_px, shift_loc_y_px, c="white",
                                     s=10, marker='X', animated=True)
                im_rm3 = ax3.scatter(shift_loc_x_px, shift_loc_y_px, c="white",
                                     s=10, marker='X', animated=True)


                tx = ax3.text(100, 115, "post", color="black", fontsize=8)
                abs_time = abs_fix_time + ms

                tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                               color="black", fontsize=8)
                ims.append([im1, im2, im3, im4, im7, im_current1,
                            im_current2, im_current3, im_rm1, im_rm3, tx, tx2])
            abs_fix_time += duration_post_ph
        # MAIN PHASE
        #durations_dummy = (None, duration_main_ph, None)
        for ms in np.arange(0, duration_main_ph, 10 / 1000):
            map_att, map_inhib, uFinal, next_fix, LL = \
                sw.evolve_maps_main((None, ms, None), fixs_x, fixs_y,
                                    map_att_prev, map_inhib_prev,
                                    fix_density_map, i_fix, sim=False)
            map_att_prev = map_att
            map_inhib_prev = map_inhib

            # plot final map
            im1 = ax1.imshow(np.float64(uFinal), animated=True,
                             origin="lower")  # , vmin=0.000061
            im2, = ax1.plot(x_path_px[range(0, i_fix)],
                            y_path_px[range(0, i_fix)],
                            c="r", animated=True)
            im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                              y_path_px[range(0, i_fix)],
                              c="r", s=3, animated=True)
            # plot inhibition map
            im4 = ax2.imshow(np.float64(map_inhib),
                             animated=True, origin="lower", cmap="YlGnBu")
            # plot attention map
            im7 = ax3.imshow(np.float64(map_att),
                             animated=True, origin="lower", cmap="YlOrRd")
            # Denote center of gaussians
            im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1],
                                      c="black", s=4, animated=True)
            im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1],
                                      c="black", s=4, animated=True)
            im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1],
                                      c="black", s=4, animated=True)
            tx = ax3.text(100, 115, "main", color="black", fontsize=8)
            abs_time = abs_fix_time + ms
            tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                           color="black", fontsize=8)
            ims.append([im1, im2, im3, im4, im7, im_current1,
                        im_current2, im_current3, tx, tx2])
        abs_fix_time += duration_main_ph
        # PRESAC PHASE
        # abs_fix_time += dur_ph1
        if not skip_pre:
            for ms in np.arange(0, duration_pre_ph, 10 / 1000):
                # get gauss centered around the upcoming location
                map_att_shift = sw.make_attention_gauss(fixs_x[1:3],
                                                        fixs_y[1:3])
                map_att_shift = sw.combine_att_fixdens(map_att_shift,
                                                       fix_density_map)
                # no locdep decay on fix nr 1
                if i_fix != 1:
                    map_att = sw.differential_time_att(ms, map_att_shift,
                                                       map_att_prev,
                                                       fixs_x=fixs_x,
                                                       fixs_y=fixs_y)
                else:
                    map_att = sw.differential_time_basic(ms, map_att_shift,
                                                         map_att_prev,
                                                         sw.omegaAttention)
                map_inhib = sw.make_inhib_gauss(fixs_x, fixs_y)
                map_inhib = sw.differential_time_basic(ms, map_inhib,
                                                       map_inhib_prev,
                                                       sw.omegaInhib)
                u = sw.combine(map_att, map_inhib)
                ustar = sw.make_positive(u)
                uFinal = sw.add_noise(ustar)
                map_inhib_prev = map_inhib
                map_att_prev = map_att

                # PLOT
                im1 = ax1.imshow(np.float64(uFinal), animated=True,
                                 origin="lower")  # , vmin=0.000061
                im2, = ax1.plot(x_path_px[range(0, i_fix)],
                                y_path_px[range(0, i_fix)],
                                c="r", animated=True)
                im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                                  y_path_px[range(0, i_fix)],
                                  c="r", s=3, animated=True)
                # plot inhibition map
                im4 = ax2.imshow(np.float64(map_inhib),
                                 animated=True, origin="lower",
                                 cmap="YlGnBu")
                # plot attention map
                im7 = ax3.imshow(np.float64(map_att),
                                 animated=True, origin="lower",
                                 cmap="YlOrRd")
                # Denote center of gaussians
                im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black", s=4,
                                          animated=True)
                im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black", s=4,
                                          animated=True)
                im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black", s=4,
                                          animated=True)

                im_next1 = ax1.scatter(x_path_px[i_fix], y_path_px[i_fix],
                                       c="white", s=4, animated=True)
                #im_next2 = ax2.scatter(x_path_px[i_fix], y_path_px[i_fix - 1],
                # c="white", s=4, animated=True)
                im_next3 = ax3.scatter(x_path_px[i_fix], y_path_px[i_fix],
                                       c="white", s=4, animated=True)

                abs_time = abs_fix_time + ms
                tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                               color="black", fontsize=8)

                tx = ax3.text(100, 115, "pre", color="black", fontsize=8)
                ims.append([im1, im2, im3, im4, im7, im_current1,
                            im_current2, im_current3, im_next1,
                            im_next3, tx, tx2])
            abs_fix_time += duration_pre_ph


        #abs_fix_time += sw.delay
        #prev_mapAtt = mapAtt_2
        #prev_mapInhib = mapInhib_2
        #abs_fix_time += duration_post_ph + duration_main_ph + duration_pre_ph
        i_fix += 1
    print("animating")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    print("saving")
    ani.save(filename)
    print("showing")
    plt.show()




def _plt_anim(ax1, ax2, ax3, uFinal, map_att, map_inhib, x_path_px,
              y_path_px, i_fix, t, ims):
    # PLOT
    im1 = ax1.imshow(np.float64(uFinal), animated=True,
                     origin="lower")  # , vmin=0.000061
    im2, = ax1.plot(x_path_px[range(0, i_fix)],
                    y_path_px[range(0, i_fix)],
                    c="r", animated=True)
    im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                      y_path_px[range(0, i_fix)],
                      c="r", s=3, animated=True)
    # plot inhibition map
    im4 = ax2.imshow(np.float64(map_inhib),
                     animated=True, origin="lower")
    # plot attention map
    im7 = ax3.imshow(np.float64(map_att),
                     animated=True, origin="lower")
    # Denote center of gaussians
    im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                              y_path_px[i_fix - 1], c="black", s=4,
                              animated=True)
    im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                              y_path_px[i_fix - 1], c="black", s=4,
                              animated=True)
    im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                              y_path_px[i_fix - 1], c="black", s=4,
                              animated=True)
    tx = ax3.text(100, 115, t, color="black", fontsize=8)
    ims.append([im1, im2, im3, im4, im7, im_current1,
                im_current2, im_current3, tx])
    return ims


def plot_dynamic_shifts_image(sw, fix_density_map, x_path, y_path, dur_path,
                              filename, image, speed=100):
    """
    Make dinamic video plot of model evolution given a scanpath when the
    presaccadic attention shift is switched on.

    Parameters
    ----------
    sw : scenewalk model object
        scenewalk model object
    fix_density_map : array
        empirical density
    x_path, y_path, dur_path : arrays
        list of datapoints, one for each fixation for x and y coordinates
        and duration
    filename : str
        where to save. must end in ".mp4"
    image : array
        base picture
    speed : int
        frames per second (?)

    Saves Result
    """
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    colors = [(0.2, 0.8, 0.1, c) for c in np.linspace(0, 1, 10)** 0.7]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

    if sw.shifts == "both":
        basic = False
    elif sw.shifts == "off":
        basic = True
    else:
        raise Exception("This model variant is not implemented")

    size1 = 14
    # initializations
    map_att_prev = sw.att_map_init()
    map_inhib_prev = sw.initialize_map_unif()

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []

    # set up figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
    ax1.set_title("Fixation Selection")
    ax2.set_title("Inhibition")
    ax3.set_title("Attention")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')

    ax1.set_xlim((0, 127))
    ax1.set_ylim((0, 127))
    ax2.set_xlim((0, 127))
    ax2.set_ylim((0, 127))
    ax3.set_xlim((0, 127))
    ax3.set_ylim((0, 127))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    abs_fix_time = 0
    abs_time = 0
    x_path_px = \
        np.asarray([sw.convert_deg_to_px(x, 'x', fix=True) for x in x_path])
    y_path_px = \
        np.asarray([sw.convert_deg_to_px(y, 'y', fix=True) for y in y_path])

    x_iter = sw.window(x_path)
    y_iter = sw.window(y_path)
    dur_iter = sw.window(dur_path)
    i_fix = 1
    # we're always evaluating the next fixation, so the last fixation's map is
    # useless
    for fixs_x, fixs_y, durs in list(zip(x_iter, y_iter, dur_iter))[0:-1]:
        print(i_fix)
        if basic:
            duration_post_ph, duration_main_ph, duration_pre_ph = \
                (0, durs[1], 0)
        else:
            duration_post_ph, duration_main_ph, duration_pre_ph = \
                sw.get_phase_times_both(i_fix, durs)
        # both mechanisms
        skip_post = True if duration_post_ph == 0 else False
        skip_pre = True if duration_pre_ph == 0 else False

        # POST PHASE
        if not skip_post:
            for ms in np.arange(0, duration_post_ph, 10 / 1000):
                map_att, shift_loc_x_px, shift_loc_y_px = \
                    sw.make_attention_gauss_post_shift(fixs_x, fixs_y,
                                                       get_loc=True)
                map_att = sw.combine_att_fixdens(map_att, fix_density_map)
                map_att = sw.differential_time_att(ms, map_att, map_att_prev,
                                                   fixs_x=fixs_x, fixs_y=fixs_y)
                map_att_prev = map_att

                map_inhib = sw.make_inhib_gauss(fixs_x, fixs_y)
                map_inhib = sw.differential_time_basic(ms, map_inhib,
                                                       map_inhib_prev,
                                                       sw.omegaInhib)
                map_inhib_prev = map_inhib

                u = sw.combine(map_att, map_inhib)
                ustar = sw.make_positive(u)
                uFinal = sw.add_noise(ustar)
                # plot final map
                im0 = ax1.imshow(image, extent=(0, 127, 0, 127))
                im1 = ax1.imshow(np.float64(uFinal), animated=True,
                                 origin="lower", cmap=cmapred, aspect=4/5)
                im2, = ax1.plot(x_path_px[range(0, i_fix)],
                                y_path_px[range(0, i_fix)],
                                c="r", animated=True)
                im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                                  y_path_px[range(0, i_fix)],
                                  c="r", s=size1, animated=True)
                # plot inhibition map
                im4 = ax2.imshow(np.float64(map_inhib),
                                 animated=True, origin="lower", cmap="YlGnBu",
                                 aspect=4/5)
                # plot attention map
                im7 = ax3.imshow(np.float64(map_att),
                                 animated=True, origin="lower", cmap="YlOrRd",
                                 aspect=4/5)
                # Denote center of gaussians
                im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)
                im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)
                im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)

                im_rm1 = ax1.scatter(shift_loc_x_px, shift_loc_y_px, c="white",
                                     s=size1, marker='X', animated=True)
                im_rm3 = ax3.scatter(shift_loc_x_px, shift_loc_y_px, c="white",
                                     s=size1, marker='X', animated=True)


                tx = ax3.text(100, 115, "post", color="black", fontsize=8)
                abs_time = abs_fix_time + ms


                tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                               color="black", fontsize=8)
                ims.append([im0, im1, im2, im3, im4, im7, im_current1,
                            im_current2, im_current3, im_rm1, im_rm3, tx, tx2])
            abs_fix_time += duration_post_ph
        # MAIN PHASE
        #durations_dummy = (None, duration_main_ph, None)
        for ms in np.arange(0, duration_main_ph, 10 / 1000):
            map_att, map_inhib, uFinal, next_fix, LL = \
                sw.evolve_maps_main((None, ms, None), fixs_x, fixs_y,
                                    map_att_prev, map_inhib_prev,
                                    fix_density_map, i_fix, sim=False)
            map_att_prev = map_att
            map_inhib_prev = map_inhib

            # plot final map
            im0 = ax1.imshow(image, extent=(0, 127, 0, 127))
            im1 = ax1.imshow(np.float64(uFinal), animated=True,
                             origin="lower", cmap=cmapred, aspect=4/5)
            im2, = ax1.plot(x_path_px[range(0, i_fix)],
                            y_path_px[range(0, i_fix)],
                            c="r", animated=True)
            im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                              y_path_px[range(0, i_fix)],
                              c="r", s=size1, animated=True)
            # plot inhibition map
            im4 = ax2.imshow(np.float64(map_inhib),
                             animated=True, origin="lower", cmap="YlGnBu",
                             aspect=4/5)
            # plot attention map
            im7 = ax3.imshow(np.float64(map_att),
                             animated=True, origin="lower", cmap="YlOrRd",
                             aspect=4/5)
            # Denote center of gaussians
            im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1], c="black", s=size1,
                                      animated=True)
            im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1],
                                      c="black", s=size1, animated=True)
            im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1],
                                      c="black", s=size1, animated=True)
            tx = ax3.text(100, 115, "main", color="black", fontsize=8)
            abs_time = abs_fix_time + ms
            tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                           color="black", fontsize=8)
            ims.append([im0, im1, im2, im3, im4, im7, im_current1,
                        im_current2, im_current3, tx, tx2])
        abs_fix_time += duration_main_ph
        # PRESAC PHASE
        # abs_fix_time += dur_ph1
        if not skip_pre:
            for ms in np.arange(0, duration_pre_ph, 10 / 1000):
                # get gauss centered around the upcoming location
                map_att_shift = sw.make_attention_gauss(fixs_x[1:3],
                                                        fixs_y[1:3])
                map_att_shift = sw.combine_att_fixdens(map_att_shift,
                                                       fix_density_map)
                # no locdep decay on fix nr 1
                if i_fix != 1:
                    map_att = sw.differential_time_att(ms, map_att_shift,
                                                       map_att_prev,
                                                       fixs_x=fixs_x,
                                                       fixs_y=fixs_y)
                else:
                    map_att = sw.differential_time_basic(ms, map_att_shift,
                                                         map_att_prev,
                                                         sw.omegaAttention)
                map_inhib = sw.make_inhib_gauss(fixs_x, fixs_y)
                map_inhib = sw.differential_time_basic(ms, map_inhib,
                                                       map_inhib_prev,
                                                       sw.omegaInhib)
                u = sw.combine(map_att, map_inhib)
                ustar = sw.make_positive(u)
                uFinal = sw.add_noise(ustar)
                map_inhib_prev = map_inhib
                map_att_prev = map_att

                # PLOT
                im0 = ax1.imshow(image, extent=(0, 127, 0, 127))
                # , vmin=0.000061
                im1 = ax1.imshow(np.float64(uFinal), animated=True,
                                 origin="lower", cmap=cmapred, aspect=4/5)
                im2, = ax1.plot(x_path_px[range(0, i_fix)],
                                y_path_px[range(0, i_fix)],
                                c="r", animated=True)
                im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                                  y_path_px[range(0, i_fix)],
                                  c="r", s=size1, animated=True)
                # plot inhibition map
                im4 = ax2.imshow(np.float64(map_inhib),
                                 animated=True, origin="lower", cmap="YlGnBu",
                                 aspect=4/5)
                # plot attention map
                im7 = ax3.imshow(np.float64(map_att),
                                 animated=True, origin="lower", cmap="YlOrRd",
                                 aspect=4/5)
                # Denote center of gaussians
                im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)
                im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)
                im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)

                im_next1 = ax1.scatter(x_path_px[i_fix], y_path_px[i_fix],
                                       c="white", s=size1, animated=True)
                im_next3 = ax3.scatter(x_path_px[i_fix], y_path_px[i_fix],
                                       c="white", s=size1, animated=True)

                abs_time = abs_fix_time + ms
                tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                               color="black", fontsize=8)

                tx = ax3.text(100, 115, "pre", color="black", fontsize=8)


                ims.append([im0, im1, im2, im3, im4, im7, im_current1,
                            im_current2, im_current3, im_next1, im_next3,
                            tx, tx2])
            abs_fix_time += duration_pre_ph


        #abs_fix_time += sw.delay
        #prev_mapAtt = mapAtt_2
        #prev_mapInhib = mapInhib_2
        #abs_fix_time += duration_post_ph + duration_main_ph + duration_pre_ph
        i_fix += 1
    print("animating")
    ani = animation.ArtistAnimation(fig, ims, interval=speed, blit=True,
                                    repeat_delay=1000)
    print("saving")
    ani.save(filename)
    print("showing")
    plt.show()







def plot_corrsac_dynamic_shifts_image(sw, fix_density_map, x_path, y_path,
                                      dur_path, filename, image, speed=100):
    """
    Make dynamic video plot of model evolution given a scanpath when the
    corrective saccade modification

    Parameters
    ----------
    sw : scenewalk model object
        scenewalk model object
    fix_density_map : array
        empirical density
    x_path, y_path, dur_path : arrays
        list of datapoints, one for each fixation for x and y coordinates
        and duration
    filename : str
        where to save. must end in ".mp4"
    image : array
        base picture
    speed : int
        frames per second (?)

    Saves Result
    """
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    colors = [(0.2, 0.8, 0.1, c) for c in np.linspace(0, 1, 10)** 0.7]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

    size1 = 14
    # initializations
    map_att_prev = sw.att_map_init()
    map_inhib_prev = sw.initialize_map_unif()

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []

    # set up figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
    ax1.set_title("Fixation Selection")
    ax2.set_title("Inhibition")
    ax3.set_title("Attention")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')

    ax1.set_xlim((0, 127))
    ax1.set_ylim((0, 127))
    ax2.set_xlim((0, 127))
    ax2.set_ylim((0, 127))
    ax3.set_xlim((0, 127))
    ax3.set_ylim((0, 127))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    abs_fix_time = 0
    abs_time = 0
    x_path_px = \
        np.asarray([sw.convert_deg_to_px(x, 'x', fix=True) for x in x_path])
    y_path_px = \
        np.asarray([sw.convert_deg_to_px(y, 'y', fix=True) for y in y_path])

    x_iter = sw.window(x_path)
    y_iter = sw.window(y_path)
    dur_iter = sw.window(dur_path)
    i_fix = 1
    # we're always evaluating the next fixation, so the last fixation's map is
    # useless
    for fixs_x, fixs_y, durs in list(zip(x_iter, y_iter, dur_iter))[0:-1]:
        print(i_fix)
        skip_pre = False
        if durs[1] > sw.tau_pre + (10/1000):
            duration_main_ph = durs[1] - sw.tau_pre
            duration_pre_ph = sw.tau_pre
        else:
            duration_main_ph = durs[1]
            skip_pre = True

        # MAIN PHASE
        #durations_dummy = (None, duration_main_ph, None)
        for ms in np.arange(0, duration_main_ph, 10 / 1000):
            durations_dummy = (None, duration_main_ph, None)
            map_att, map_inhib, uFinal, next_fix, LL = \
                sw.evolve_maps_main(durations_dummy, fixs_x, fixs_y,
                                    map_att_prev, map_inhib_prev,
                                    fix_density_map, i_fix, sim=False)
            map_att_prev = map_att
            map_inhib_prev = map_inhib

            # plot final map #vmin=0.000061
            im0 = ax1.imshow(image, extent=(0, 127, 0, 127))
            im1 = ax1.imshow(np.float64(uFinal), animated=True,
                             origin="lower", cmap=cmapred, aspect=4/5)
            im2, = ax1.plot(x_path_px[range(0, i_fix)],
                            y_path_px[range(0, i_fix)],
                            c="r", animated=True)
            im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                              y_path_px[range(0, i_fix)],
                              c="r", s=size1, animated=True)
            # plot inhibition map
            im4 = ax2.imshow(np.float64(map_inhib),
                             animated=True, origin="lower", cmap="YlGnBu",
                             aspect=4/5)
            # plot attention map
            im7 = ax3.imshow(np.float64(map_att),
                             animated=True, origin="lower", cmap="YlOrRd",
                             aspect=4/5)
            # Denote center of gaussians
            im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1], c="black", s=size1,
                                      animated=True)
            im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1], c="black", s=size1,
                                      animated=True)
            im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                      y_path_px[i_fix - 1], c="black", s=size1,
                                      animated=True)
            tx = ax3.text(100, 115, "main", color="black", fontsize=8)
            abs_time = abs_fix_time + ms
            tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                           color="black", fontsize=8)
            ims.append([im0, im1, im2, im3, im4, im7, im_current1,
                        im_current2, im_current3, tx, tx2])
        abs_fix_time += duration_main_ph
        # PRESAC PHASE
        # abs_fix_time += dur_ph1

        landed_fix = next_fix
        u, mag = sw.get_unit_vector([fixs_x[1], fixs_y[1]], next_fix)
        shift_by = mag * sw.shift_size
        aimed_fix_x = landed_fix[0]+(u[0]*shift_by)
        aimed_fix_y = landed_fix[1]+(u[1]*shift_by)

        if not skip_pre:
            for ms in np.arange(0, duration_pre_ph, 10 / 1000):
                # get gauss centered around aimed for location
                # we give it the current to next fixes (we want the location
                # shifted for the vector current (1)-> next (2))
                # its going to make you a gaussian out from the next, which is
                # what we are "aiming for" with the saccade
                map_att_shift = \
                    sw.make_attention_gauss_pre([None, aimed_fix_x, None],
                                                [None, aimed_fix_y, None])
                map_att_shift = sw.combine_att_fixdens(map_att_shift,
                                                       fix_density_map)
                # no location dependent decay on fixation 1
                if i_fix != 1:
                    map_att_pre = sw.differential_time_att(duration_pre_ph,
                                                           map_att_shift,
                                                           map_att_prev,
                                                           fixs_x=fixs_x,
                                                           fixs_y=fixs_y)
                else:
                    map_att_pre = sw.differential_time_basic(duration_pre_ph,
                                                             map_att_shift,
                                                             map_att_prev,
                                                             sw.omegaAttention)

                map_inhib_pre = sw.make_inhib_gauss(fixs_x, fixs_y)
                map_inhib_pre = sw.differential_time_basic(duration_pre_ph,
                                                           map_inhib_pre,
                                                           map_inhib_prev,
                                                           sw.omegaInhib)
                u = sw.combine(map_att_pre, map_inhib_pre)
                ustar = sw.make_positive(u)
                uFinal_pre = sw.add_noise(ustar)
                map_att_prev = map_att_pre
                map_inhib_prev = map_inhib_pre


                # PLOT
                im0 = ax1.imshow(image, extent=(0, 127, 0, 127))
                im1 = ax1.imshow(np.float64(uFinal_pre), animated=True,
                                 origin="lower", cmap=cmapred, aspect=4/5)
                im2, = ax1.plot(x_path_px[range(0, i_fix)],
                                y_path_px[range(0, i_fix)],
                                c="r", animated=True)
                im3 = ax1.scatter(x_path_px[range(0, i_fix)],
                                  y_path_px[range(0, i_fix)],
                                  c="r", s=size1, animated=True)
                # plot inhibition map
                im4 = ax2.imshow(np.float64(map_inhib_pre),
                                 animated=True, origin="lower",
                                 cmap="YlGnBu", aspect=4/5)
                # plot attention map
                im7 = ax3.imshow(np.float64(map_att_pre),
                                 animated=True, origin="lower",
                                 cmap="YlOrRd", aspect=4/5)
                # Denote center of gaussians
                im_current1 = ax1.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)
                im_current2 = ax2.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)
                im_current3 = ax3.scatter(x_path_px[i_fix - 1],
                                          y_path_px[i_fix - 1], c="black",
                                          s=size1, animated=True)


                aim_px_x = sw.convert_deg_to_px(aimed_fix_x, 'x', fix=True)
                aim_px_y = sw.convert_deg_to_px(aimed_fix_y, 'y', fix=True)
                im_next1 = ax1.scatter(x_path_px[i_fix], y_path_px[i_fix],
                                       c="white", s=size1, animated=True)
                #im_next2 = ax2.scatter(x_path_px[i_fix], y_path_px[i_fix - 1],
                # c="white", s=4, animated=True)
                im_next3 = ax3.scatter(x_path_px[i_fix], y_path_px[i_fix],
                                       c="white", s=size1, animated=True)

                im_rm1 = ax1.scatter(aim_px_x, aim_px_y, c="red", s=size1,
                                     animated=True)
                im_rm2 = ax3.scatter(aim_px_x, aim_px_y, c="red", s=size1,
                                     animated=True)


                abs_time = abs_fix_time + ms
                tx2 = ax3.text(95, 100, "{0:.2f} s".format(abs_time),
                               color="black", fontsize=8)

                tx = ax3.text(100, 115, "pre", color="black", fontsize=8)


                ims.append([im0, im1, im2, im3, im4, im7, im_current1, im_rm1,
                            im_rm2, im_current2, im_current3, im_next1,
                            im_next3, tx, tx2])
            abs_fix_time += duration_pre_ph


        #abs_fix_time += sw.delay
        #prev_mapAtt = mapAtt_2
        #prev_mapInhib = mapInhib_2
        #abs_fix_time += duration_post_ph + duration_main_ph + duration_pre_ph
        i_fix += 1
    print("animating")
    ani = animation.ArtistAnimation(fig, ims, interval=speed, blit=True,
                                    repeat_delay=1000)
    print("saving")
    ani.save(filename)
    print("showing")
    plt.show()
