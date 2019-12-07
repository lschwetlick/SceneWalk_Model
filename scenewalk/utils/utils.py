""" Utility Functions for SceneWalk Related stuff """

def get_git_sha():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

def trpd(my_mean, my_std, lb, ub):
    from scipy.stats import norm as normal, truncnorm as truncated_normal
    a, b = (lb - my_mean) / my_std, (ub - my_mean) / my_std
    return truncated_normal(a=a, b=b, scale=my_std, loc=my_mean)

def save2dict_by_subj(chains_list, all_vp_list, def_args, fname, perc_last_samples=100):
    """
    saves full estimation results as a dictionary
    """
    from collections import OrderedDict
    import numpy as np

    vp_params = OrderedDict({})
    for chains, vp in list(zip(chains_list, all_vp_list)):
        if chains is None:
            continue
        param_dict1 = OrderedDict({})
        param_ix = 0
        for param_name in list(def_args.keys()):
            if def_args[param_name] is None:
                chain_len = chains.shape[1]
                samp_ix =  int(chain_len - (chain_len * perc_last_samples/100))
                chain1 = chains[0, samp_ix:, param_ix]
                chain2 = chains[1, samp_ix:, param_ix]
                chain3 = chains[2, samp_ix:, param_ix]
                allchains = np.hstack((chain1, chain2, chain3))
                param_dict1[param_name] = allchains
                param_ix += 1
            else:
                param_dict1[param_name] = def_args[param_name]

        vp_params[vp] = param_dict1
    np.save(fname, vp_params)
    return (vp_params)



def save2npy_point_estimate_by_subj(chains_list, all_vp_list, def_args, alpha, fname, CI=False, perc_last_samples=75):
    from collections import OrderedDict
    import numpy as np
    from pymc3.stats import hpd

    vp_params = OrderedDict({})
    for chains, vp in list(zip(chains_list, all_vp_list)):
        if chains is None:
            continue
        param_dict1 = OrderedDict({})
        param_ix = 0
        for param_name in list(def_args.keys()):
            if def_args[param_name] is None:
                chain_len = chains.shape[1]
                samp_ix =  int(chain_len - (chain_len * perc_last_samples/100))
                chain1 = chains[0, samp_ix:, param_ix]
                chain2 = chains[1, samp_ix:, param_ix]
                chain3 = chains[2, samp_ix:, param_ix]
                allchains = np.hstack((chain1, chain2, chain3))

                # Highst Posterior Density
                hpd_all = hpd(allchains, alpha)
                mpde = (hpd_all[1]+hpd_all[0])/2
                if CI:
                    param_dict1[param_name] = [mpde, hpd_all[1] - mpde]
                else:
                    param_dict1[param_name] = mpde
                param_ix += 1
            else:
                param_dict1[param_name] = def_args[param_name]

        vp_params[vp] = param_dict1
    np.save(fname, vp_params)
    return (vp_params)

def save2pd_overall_point_estimates(chains_list, all_vp_list, def_args, priors, sw, alpha, fname, perc_last_samples=75):
    from collections import OrderedDict
    import numpy as np
    from pymc3.stats import hpd
    import pandas as pd

    param_ix = 0
    rows_list = []
    for param_name in def_args.keys():
        if param_name in list(priors.keys()):
            print(param_name)
            allvps = []
            for vp in all_vp_list:
                #print(vp)
                chains = chains_list[vp]
                if chains is None:
                    continue
                chain_len = chains.shape[1]
                samp_ix =  int(chain_len - (chain_len * perc_last_samples/100))
                chain1 = chains[0, samp_ix:, param_ix]
                chain2 = chains[1, samp_ix:, param_ix]
                chain3 = chains[2, samp_ix:, param_ix]
                allvps.extend(chain1)
                allvps.extend(chain2)
                allvps.extend(chain3)
            allvps = np.array(allvps)
            #print(len(allvps))
            hpd_all = hpd(allvps, alpha)
            mpde = (hpd_all[1]+hpd_all[0])/2
            #print(hpd_all)
            #break
            dict1 = {"param_name":param_name, "mpde":mpde, "interv":mpde-hpd_all[0], "left": hpd_all[0], "right": hpd_all[1]}
            param_ix+=1
        else:
            dict1 = {"param_name":param_name, "mpde":def_args[param_name], "interv":np.nan, "left": np.nan, "right": np.nan}
        rows_list.append(dict1)
    rows_list.append({"param_name":"tau_pre", "mpde":sw.tau_pre, "interv":np.nan, "left": np.nan, "right": np.nan})
    rows_list.append({"param_name":"tau_post", "mpde":sw.tau_post, "interv":np.nan, "left": np.nan, "right": np.nan})
    rows_list.append({"param_name":"foR_size", "mpde":sw.foR_size, "interv":np.nan, "left": np.nan, "right": np.nan})
    hpde_df = pd.DataFrame(rows_list)
    hpde_df.to_csv(fname)
    return(hpde_df)

def save2pd_subj_point_estimates(chains_list, all_vp_list, priors, alpha, fname, perc_last_samples=75):
    from collections import OrderedDict
    import numpy as np
    from pymc3.stats import hpd
    import pandas as pd
    rows_list = []
    #vp_id = 0
    for vp in all_vp_list:
        chains = chains_list[vp]
        if chains is None:
            continue
        tmp_df = pd.DataFrame()
        param_ix = 0
        for param_name in list(priors.keys()):
            chain_len = chains.shape[1]
            samp_ix =  int(chain_len - (chain_len * perc_last_samples/100))
            chain1 = chains[0, samp_ix:, param_ix]
            chain2 = chains[1, samp_ix:, param_ix]
            chain3 = chains[2, samp_ix:, param_ix]
            allchains = []
            allchains = np.hstack((chain1, chain2, chain3))
            hpd_all = hpd(allchains, alpha)
            mpde = (hpd_all[1]+hpd_all[0])/2

            dict1 = {"vp": vp, "param_name":param_name, "mpde":mpde, "interv":mpde-hpd_all[0], "left": hpd_all[0], "right": hpd_all[1]}
            rows_list.append(dict1)

        param_ix += 1
        tmp_df['vp'] = vp
    # vp_id +=1
    hpde_df = pd.DataFrame(rows_list)
    hpde_df.to_csv(fname)
    return(hpde_df)


def get_all_colors():
    import matplotlib.colors as mplcolor
    cols = list(mplcolor.cnames.keys())
    return cols

def show_all_colors():
    import matplotlib.patches as mpatches
    from matplotlib import pyplot as plt
    cols = get_all_colors()
    fig, ax = plt.subplots(figsize=(3, 43))
    pat_list = []
    for c in cols:
        pat = mpatches.Patch(color=c, label=c)
        pat_list.append(pat)
    plt.legend(handles=pat_list)
    return fig, ax
