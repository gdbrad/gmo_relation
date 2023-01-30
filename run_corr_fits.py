import h5py as h5
#from h5glance import H5Glance
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import gvar as gv
import pandas as pd
import os 
import sys
import ipywidgets as widgets
import matplotlib.pyplot as plt
# %matplotlib notebook
#import h5ls
import lsqfit
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from pathlib import Path
import importlib
import argparse
import platform

import gmo_fitter as gmo 
import i_o 
import gmo_fit_analysis as fa
import load_data_priors as ld
import mass_relations as ma
import delta_gmo_fit as gmo_xpt 
matplotlib.rcParams['figure.figsize'] = [10, 8]

# def plot_m4(results):


def main():
    parser = argparse.ArgumentParser(description='analysis of simult. fit to the 4 baryons that form the gmo relation, also fit the gmo product correlator directly')
    parser.add_argument('fit_params', help='input file to specify fit')
    parser.add_argument('fit_type',help='specify simultaneous baryon fit with or without gmo product correlator as input')
    parser.add_argument('pdf',help='generate a pdf and output plot?',default=True)
    # parser.add_argument('xpt',help='run gmo xpt analysis?',default=True)

    args = parser.parse_args()
    # if args.save_figs and not os.path.exists('figures'):
    #     os.makedirs('figures')
    print(args)
    # add path to the input file and load it
    sys.path.append(os.path.dirname(os.path.abspath(args.fit_params)))
    fp = importlib.import_module(
        args.fit_params.split('/')[-1].split('.py')[0])

    
    if platform.system() == 'Darwin':
        file = '/Users/grantdb/lqcd/data/c51_2pt_octet_decuplet.h5'
    else:
        file = '/home/gmoney/lqcd/data/c51_2pt_octet_decuplet.h5'
    with h5.File(file,"r") as f:
        ensembles = {}
        ensembles = list(f.keys())

    p_dict = fp.p_dict
    abbr = p_dict['abbr']

    nucleon_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='proton')
    lam_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='lambda_z')
    xi_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='xi_z')
    sigma_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='sigma_p')
    piplus_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='piplus')
    kplus_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='kplus')
    delta_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='delta_pp')
    gmo_ratio_raw = ld.G_gmo(file,p_dict['abbr'],log=False)


    prior_nucl = {}
    prior = {}
    states=p_dict['gmo_states_all']
    newlist = [x for x in states]
    for x in newlist:
        path = os.path.normpath("./priors/{0}/{1}/prior_nucl.csv".format(p_dict['abbr'],x))
        df = pd.read_csv(path, index_col=0).to_dict()
        for key in list(df.keys()):
            length = int(np.sqrt(len(list(df[key].values()))))
            prior_nucl[key] = list(df[key].values())[:length]
            # prior_nucl['gmo_E'] = list([np.repeat(gv.gvar('0.0030(27)'),8)])
        prior = gv.gvar(prior_nucl)

    if args.fit_type == 'simult_baryons':
        model_type = 'simult_baryons'
        gmo_ = fa.fit_ensemble(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],prior=prior,
        nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,sigma_corr_data=sigma_corr,delta_corr_data=None,gmo_corr_data=None,
        piplus_corr_data=None,kplus_corr_data=None,model_type=model_type)
        # print(gmo_.get_fit().format(maxline=True))
        print(gmo_)
        # print(gmo_.get_m_4(t=None))
        fit_out = gmo_.get_fit()
        gmo_.plot_effective_mass( t_plot_min=0,t_plot_max=15, model_type=model_type, fig_name='plots/{0}_{1}'.format(abbr,model_type),show_fit=True)
        out_path = 'fit_results/{0}/'.format(p_dict['abbr'],model_type)

        if os.path.exists(out_path):
            pass
        else:
            os.mkdir(out_path)

        if args.pdf:
            plots = gmo_.make_plots(model_type=model_type, fig_name='plots/{0}_{1}'.format(abbr,model_type))
            output_dir = 'fit_results/{0}/{1}_{0}.pdf'.format(p_dict['abbr'],model_type)
            output_pdf = PdfPages(output_dir)
            for plot in plots:
                if plot is not None:
                    output_pdf.savefig(plot)
            # if p_dict['show_many_states']:
            #     output_pdf.savefig(fit_ensemble.plot_stability(model_type='corr', n_states_array=[1, 2, 3, 4]))
            output_pdf.close()

    if args.fit_type == 'mesons':
        model_type ='mesons'


    if args.fit_type == 'simult_baryons_gmo':
        model_type = 'simult_baryons_gmo'
        gmo_ = fa.fit_ensemble(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],prior=prior,
        nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,sigma_corr_data=sigma_corr,delta_corr_data=None,gmo_corr_data=gmo_ratio_raw,
        piplus_corr_data=None,kplus_corr_data=None,model_type=model_type)
        print(gmo_)
        fit_out= gmo_.get_fit()

        out_path = 'fit_results/{0}_{1}'.format(p_dict['abbr'],model_type)
        if os.path.exists(out_path):
            pass
        else:
            os.mkdir(out_path)

        ld.pickle_out(fit_out=fit_out,out_path=out_path,species="baryon_w_gmo")

        if args.pdf:
            plots = gmo_.make_gmo_plots(model_type='simult_baryons_gmo', fig_name='plots/{0}_{1}'.format(abbr,model_type))
            output_dir = 'fit_results/{0}/{1}_{0}.pdf'.format(p_dict['abbr'],model_type)
            output_pdf = PdfPages(output_dir)
            for plot in plots:
                if plot is not None:
                    output_pdf.savefig(plot)
            # if p_dict['show_many_states']:
            #     output_pdf.savefig(fit_ensemble.plot_stability(model_type='corr', n_states_array=[1, 2, 3, 4]))
            output_pdf.close()


    ''' xpt routines 
    '''
    # if args.xpt:
    #     model_info = fp.model_info



    



if __name__ == "__main__":
    main()