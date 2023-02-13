import gvar as gv 
import h5py as h5 
import numpy as np 
import os 

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
            data['SS'] = f[particle_path][:, :, 0, 0]
            data['PS'] = f[particle_path][:, :, 1, 0] 
    return data

def G_gmo(file_h5,abbr):
    temp = {}
    for smr in get_raw_corr(file_h5=file_h5, abbr=abbr,particle='proton'): 
        for part in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
            temp[(part, smr)] = get_raw_corr(file_h5=file_h5, abbr=abbr,particle=part)[smr]
    temp = gv.dataset.avg_data(temp)

    print(temp)
    output = {}
    for smr in get_raw_corr(file_h5=file_h5, abbr=abbr,particle='proton'):
            output[smr] = (
                temp[('lambda_z', smr)]
                * np.power(temp[('sigma_p', smr)], 1/3)
                * np.power(temp[('proton', smr)], -2/3)
                * np.power(temp[('xi_z', smr)], -2/3)
            )
    return output
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
