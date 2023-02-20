import gvar as gv 
import h5py as h5 
import numpy as np 
import os 
import bs_utils as bs 
from _collections_abc import MutableMapping 
def pickle_out(fit_out,out_path,species=None):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fit_dump = {}
    fit_dump['prior'] = fit_out.prior
    fit_dump['p'] = fit_out.p
    fit_dump['logGBF'] = fit_out.logGBF
    fit_dump['Q'] = fit_out.Q
    if species == 'meson':
        return gv.dump(fit_dump,out_path+'meson_fit_params')
    elif species == 'baryon':
        return gv.dump(fit_dump,out_path+'fit_params')
    elif species == 'baryon_w_gmo':
        return gv.dump(fit_dump,out_path+'fit_params_all')

def print_posterior(out_path):
    posterior = {}
    post_out = gv.load(out_path+"fit_params")
    posterior['lam_E0'] = post_out['p']['lam_E0']
    posterior['lam_E1'] = np.exp(post_out['p']['lam_log(dE)'][0])+posterior['lam_E0']
    posterior['proton_E0'] = post_out['p']['proton_E0']
    posterior['proton_E1'] = np.exp(post_out['p']['proton_log(dE)'][0])+posterior['proton_E0']
    posterior['sigma_E0'] = post_out['p']['sigma_E0']
    posterior['sigma_E1'] = np.exp(post_out['p']['sigma_log(dE)'][0]) + posterior['sigma_E0']
    posterior['xi_E0'] = post_out['p']['xi_E0']
    posterior['xi_E1'] = np.exp(post_out['p']['xi_log(dE)'][0]) + posterior['xi_E0']

    return posterior
def get_raw_corr(file_h5,abbr,particle):
    data = {}
    particle_path = '/'+abbr+'/'+particle
    with h5.File(file_h5,"r") as f:
        if f[particle_path].shape[3] == 1:
            data['SS'] = f[particle_path][:, :, 0, 0].real
            data['PS'] = f[particle_path][:, :, 1, 0].real 
    return data

def G_gmo(file_h5,abbr):
    temp = {}
    for smr in get_raw_corr(file_h5=file_h5, abbr=abbr,particle='proton'): 
        for part in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
            temp[(part, smr)] = get_raw_corr(file_h5=file_h5, abbr=abbr,particle=part)[smr]
    temp = gv.dataset.avg_data(temp)

    output = {}
    for smr in get_raw_corr(file_h5=file_h5, abbr=abbr,particle='proton'):
            output[smr] = (
                temp[('lambda_z', smr)]
                * np.power(temp[('sigma_p', smr)], 1/3)
                * np.power(temp[('proton', smr)], -2/3)
                * np.power(temp[('xi_z', smr)], -2/3)
            )
    return output

def get_raw_corr_new(file_h5,abbr):
    data = {}
    
    with h5.File(file_h5,"r") as f:
        for baryon in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
            particle_path = '/'+abbr+'/'+baryon
            data[baryon+'_SS'] = f[particle_path][:, :, 0, 0].real
            data[baryon+'_PS'] = f[particle_path][:, :, 1, 0].real 
    return data

def G_gmo_bs(file_h5,abbr,bsN,bs_list):
    data = {}
    gmo = {}
    for src_snk in ['PS', 'SS']:
        gmo['gmo_'+src_snk] = {}
        for baryon in['lambda_z', 'sigma_p', 'proton', 'xi_z']:
            data[baryon+'_'+src_snk] = np.zeros(bsN)
    temp = {}
    temp = get_raw_corr_new(file_h5=file_h5, abbr=abbr)
    for n in range(bsN):
        corr_bs_copy = resample_correlator(temp,bs_list=bs_list,n=n)
        for src_snk in ['PS', 'SS']:
            for baryon in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
                data[baryon+'_'+src_snk][n] = np.mean(corr_bs_copy[baryon+'_'+src_snk], axis=(0,1),dtype=object)
        for src_snk in ['PS', 'SS']:
            gmo['gmo_'+src_snk][str(n)] = data['lambda_z_'+src_snk][n] *np.power(data['sigma_p_'+src_snk][n], 1/3) *np.power(data['proton_'+src_snk][n], -2/3) * np.power(data['xi_z_'+src_snk][n], -2/3)
    # hacky way to eliminate the indexing done above, there is certainly a cleaner way to do #
            Samples_ps= gmo['gmo_PS']
            Samples_ss= gmo['gmo_SS']
            vals_ps = np.fromiter(Samples_ps.values(), dtype=float)
            vals_ss = np.fromiter(Samples_ss.values(), dtype=float)
            gmo_bs = {}
            gmo_bs['gmo_PS'] = vals_ps
            gmo_bs['gmo_SS'] = vals_ss
    print(gmo_bs)
        # print(gmo_bs)
    correlators = gv.dataset.avg_data(gmo_bs, bstrap=True)

    return correlators


def delete_keys_from_dict(dictionary, keys):
    keys_set = set(keys)  # Just an optimization for the "if key in keys" lookup.

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, MutableMapping):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)
            else:
                modified_dict[key] = value  # or copy.deepcopy(value) if a copy is desired for non-dicts.
    return modified_dict

def resample_correlator(raw_corr,bs_list, n):
    resampled_raw_corr_data = ({key : raw_corr[key][bs_list[n, :], :]
    for key in raw_corr.keys()})
    resampled_corr_gv = resampled_raw_corr_data
    return resampled_corr_gv


# def G_gmo(file_h5,abbr,bsN=None):
#     if bsN is None:
#         bsN = 50
#     data = {}
#     for src_snk in ['PS', 'SS']:
#             for baryon in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
#                 data[baryon+'_'+src_snk] = np.zeros(bsN)

#     for n in range(bsN):
#         corr_bs_copy = make_bootstrap_copy(raw_correlators, n) # a dictionary with entries for each octet baryon and src/smr combination, eg key 'nucleon_PS'
#         for src_snk in ['PS', 'SS']:
#             for baryon in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
#             data['gmo_'+src_snk][n] = (
#                 data['lambda_z_'+src_snk][n] 
#                 * np.power(data['sigma_p_'+src_snk][n], 1/3) 
#                 * np.power(data['proton_'+src_snk][n],-2/3)
#                 * np.power(data['xi_z'+src_snk][n], -2/3)
#             )

#     correlators = gv.dataset.avg_data(data, bstrap=True)
def fetch_prior(p_dict,states):

    prior_nucl = {}
    newlist = [x for x in states]
    for i,x in enumerate(newlist):
        path = os.path.normpath("./priors/{0}/{1}/prior_nucl.csv".format(p_dict['abbr'],x))
        df = pd.read_csv(path, index_col=0).to_dict()
        for key in list(df.keys()):
            length = int(np.sqrt(len(list(df[key].values()))))
            prior_nucl[key] = list(df[key].values())[:length]
        prior = {}
        prior[x] = {}
        prior[x] = gv.gvar(prior_nucl)
        print(prior)
        # prior = {**prior_nucl}
        # prior_nucl = gv.BufferDict()
        # prior[x] = prior_nucl
    return prior
