from math import isnan
import tqdm
import h5py as h5
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import gvar as gv
import pandas as pd
import os 
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import importlib
import argparse
import platform

import bs_utils as bs 
import i_o 
import gmo_fit_analysis as fa
import load_data_priors as ld
import mass_relations as ma
import delta_gmo_fit as gmo_xpt 
import gmo_fitter as fitter
matplotlib.rcParams['figure.figsize'] = [10, 8]

importlib.reload(fa)


def main():
    parser = argparse.ArgumentParser(description='analysis of simult. fit to the 4 baryons that form the gmo relation, also fit the gmo product correlator directly')
    parser.add_argument('fit_params', help='input file to specify fit')
    parser.add_argument('fit_type',help='specify simultaneous baryon fit with or without gmo product correlator as input')
    parser.add_argument('pdf',help='generate a pdf and output plot?',default=True)
    parser.add_argument('--bs',help='perform bootstrapping?',default=False, action='store_true') 
    parser.add_argument('--bsn',help='number of bs samples',type=int,default=2000) 

    # parser.add_argument('xpt',help='run gmo xpt analysis?',default=True)

    args = parser.parse_args()
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
    if abbr  == 'a12m180S' or abbr == 'a12m220':
        nucleon_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='proton')
        prior_nucl = {}
        prior = {}
        states=p_dict['states']
        newlist = [x for x in states]
        for x in newlist:
            path = os.path.normpath("./priors/{0}/{1}/prior_nucl.csv".format(p_dict['abbr'],x))
            df = pd.read_csv(path, index_col=0).to_dict()
            for key in list(df.keys()):
                length = int(np.sqrt(len(list(df[key].values()))))
                prior_nucl[key] = list(df[key].values())[:length]
                # prior_nucl['gmo_E'] = list([np.repeat(gv.gvar('0.0030(27)'),8)])
            prior = gv.gvar(prior_nucl)

    # pull in raw corr data
    raw_corr = {}
    nucleon_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='proton')
    lam_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='lambda_z')
    xi_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='xi_z')
    sigma_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='sigma_p')
    gmo_ratio_raw = ld.G_gmo(file,p_dict['abbr'])
    ncfg = xi_corr['PS'].shape[0]
    raw_corr['proton'] = nucleon_corr
    raw_corr['lam'] = lam_corr
    raw_corr['sigma'] = sigma_corr
    raw_corr['xi'] = xi_corr
    raw_corr['gmo_ratio'] = gmo_ratio_raw

    if args.fit_type == 'simult_baryons':
        model_type = args.fit_type
        prior = ld.fetch_prior(model_type,p_dict)
        sim_baryons = fa.fit_analysis(t_range=p_dict
        ['t_range'],simult=True,t_period=64,states=p_dict['gmo_states'],p_dict=p_dict, n_states=p_dict['n_states'],prior=prior,
        nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,sigma_corr_data=sigma_corr,
        gmo_corr_data=None,model_type=model_type)
        print(sim_baryons)
        fit_out = sim_baryons.get_fit()
        
        out_path = 'fit_results/{0}/{1}/'.format(p_dict['abbr'],model_type)

        ld.pickle_out(fit_out=fit_out,out_path=out_path,species="baryon")
        print(ld.print_posterior(out_path=out_path))
        if args.pdf:
            plot1 = sim_baryons.return_best_fit_info()
            plot2 = sim_baryons.plot_effective_mass(t_plot_min=0, t_plot_max=40,model_type=model_type, 
            show_plot=True,show_fit=True)
            plot3 = sim_baryons.plot_effective_wf(model_type=model_type, t_plot_min=0, t_plot_max=40, 
            show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
                # pp.savefig(plot3)

            output_pdf.close()
    
    # 1. perform fits to individual baryon correlators #


    elif args.fit_type == 'gmo_direct':
        model_type = args.fit_type
        prior = ld.fetch_prior(model_type,p_dict)
        gmo_direct = fa.fit_analysis(t_range=p_dict
        ['t_range'],simult=False,t_period=64,states=p_dict['gmo_direct'],p_dict=p_dict, 
        n_states=p_dict['n_states'],prior=prior,
        nucleon_corr_data=None,lam_corr_data=None, xi_corr_data=None,sigma_corr_data=None,
        gmo_corr_data=gmo_ratio_raw,model_type=model_type)
        print(gmo_direct)
        fit_out = gmo_direct.get_fit()
        gmo_eff_mass = gmo_direct.get_gmo_effective(gmo_ratio=gmo_ratio_raw)
        if args.pdf:
            plot1 = gmo_direct.return_best_fit_info(bs=False)
            plot2 = gmo_direct.plot_delta_gmo(correlators_gv=gmo_ratio_raw,t_plot_min=0,t_plot_max=20,
            model_type=model_type,fig_name = None,show_fit=True)
            plot3 = gmo_direct.plot_gmo_effective_mass(effective_mass=gmo_eff_mass,
            t_plot_min=0,t_plot_max=40,model_type=model_type,show_fit=True,fig_name=None)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
                pp.savefig(plot3)
        if args.bs:
            bsN = args.bsn
            ncfg = gmo_ratio_raw['PS'].shape[0]
            bs_list = bs.get_bs_list(Ndata=ncfg,Nbs=bsN)
            prelim_fit_keys = sorted(fit_out.p.keys()) 
            output = {key : [] for key in prelim_fit_keys}
            gmo_bs = ld.G_gmo_bs(file,p_dict['abbr'],bsN=bsN,bs_list=bs_list)
            def resample_corr(bs_list,n):
                resampled_raw_corr_data = ({key : gmo_bs[key][bs_list[n, :]]
                for key in gmo_bs.keys()})
                resampled_corr_gv = gv.dataset.avg_data(resampled_raw_corr_data)
                return resampled_corr_gv

            for j in tqdm.tqdm(range(bsN), desc='bootstrap'):
                gv.switch_gvar() 
                
                temp_gmo_corr = resample_corr(bs_list,j)
                print(temp_gmo_corr)
                # make fit with resampled correlators
                gmo_bs_out = fitter.fitter(t_range=p_dict
                ['t_range'],t_period=64,simult=False, p_dict=p_dict, states= p_dict['gmo_direct'],n_states=p_dict['n_states'],prior=prior,
                nucleon_corr=None,lam_corr=None, xi_corr=None,
                sigma_corr=None,gmo_ratio_corr=temp_gmo_corr,model_type=model_type)
                temp_fit = gmo_bs_out.get_fit()
                print(temp_fit)

                for key in prelim_fit_keys:
                    p = temp_fit.pmean[key]
                    output[key].append(p)

                gv.restore_gvar()

                # print results -- should be similar to previous results
            table = gv.dataset.avg_data(output, bstrap=True)
            print(gv.tabulate(table))
        #     print(gmo_bs)
        #     gmo_direct_bs = fa.fit_analysis(t_range=p_dict
        #     ['t_range'],simult=False,t_period=64,states=p_dict['gmo_direct'],p_dict=p_dict, 
        #     n_states=p_dict['n_states'],prior=prior,
        #     nucleon_corr_data=None,lam_corr_data=None, xi_corr_data=None,sigma_corr_data=None,
        #     gmo_corr_data=gmo_bs,model_type=model_type)
        #     print(gmo_direct_bs)
        #     fit_out = gmo_direct_bs.get_fit()
        # else:
            

        




            # output_pdf.close()
    elif args.fit_type == 'xi':
        prior_xi = {k:v for k,v in prior.items() if 'xi' in k}
        # print(new_d)
        model_type = 'xi'
        xi_ = fa.fit_analysis(t_range=p_dict
        ['t_range'],p_dict=p_dict,simult=False,states=['xi'], t_period=64, n_states=p_dict['n_states'],prior=prior_xi,
        nucleon_corr_data=None,lam_corr_data=None, xi_corr_data=xi_corr,
        sigma_corr_data=None,gmo_corr_data=None,model_type=model_type)
        # print(xi_)
        fit_out = xi_.get_fit()
        print(fit_out.formatall(maxline=True))
        if args.pdf:
            # plot1 = xi_.return_best_fit_info()
            plot2 = xi_.plot_effective_mass(t_plot_min=5, t_plot_max=30, model_type = model_type,show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                # pp.savefig(plot1)
                pp.savefig(plot2)
            output_pdf.close()

    elif args.fit_type == 'lam':
        prior_lam = {k:v for k,v in prior.items() if 'lam' in k}
        # print(new_d)
        model_type = 'lam'
        lam_ = fitter.fitter(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],states=['lam'],prior=prior_lam,
        nucleon_corr=None,lam_corr=gv.dataset.avg_data(lam_corr), xi_corr=None,
        sigma_corr=None,delta_corr=None,gmo_ratio_corr=None,
        piplus_corr=None,kplus_corr=None,model_type=model_type)
        # print(xi_)
        fit_out = lam_.get_fit()
        print(fit_out.formatall(maxline=True))
        if args.pdf:
            plot1 = lam_.return_best_fit_info()
            plot2 = lam_.plot_effective_mass(tmin=None, tmax=None, ylim=None, show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
            output_pdf.close()

    elif args.fit_type == 'proton':
        prior_proton = {k:v for k,v in prior.items() if 'proton' in k}
        # print(new_d)
        model_type = 'proton'
        proton_ = fitter.fitter(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],states=['proton'],prior=prior_proton,
        nucleon_corr=gv.dataset.avg_data(nucleon_corr),lam_corr=None, xi_corr=None,
        sigma_corr=None,gmo_ratio_corr=None,model_type=model_type)
        fit_out = proton_.get_fit()
        print(fit_out.formatall(maxline=True))
        print(str(np.exp(fit_out.p['proton_log(dE)'][0])))
        if args.pdf:
            plot1 = proton_.return_best_fit_info()
            plot2 = proton_.plot_effective_mass(tmin=None, tmax=None, ylim=None, show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
            # output_pdf.close()

    elif args.fit_type == 'sigma':
        prior_sigma = {k:v for k,v in prior.items() if 'sigma' in k}
        # print(new_d)
        model_type = 'sigma'
        sigma_ = fitter.fitter(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],states=['sigma'],prior=prior_sigma,
        nucleon_corr=None,lam_corr=None, xi_corr=None,
        sigma_corr=gv.dataset.avg_data(sigma_corr),delta_corr=None,gmo_ratio_corr=None,
        piplus_corr=None,kplus_corr=None,model_type=model_type)
        # print(xi_)
        fit_out = sigma_.get_fit()
        print(fit_out.formatall(maxline=True))
        if args.pdf:
            plot1 = sigma_.return_best_fit_info()
            plot2 = sigma_.plot_effective_mass(tmin=None, tmax=None, ylim=None, show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
            output_pdf.close()

    elif args.fit_type == 'simult_baryons_gmo':
        model_type = args.fit_type
        prior = ld.fetch_prior(model_type,p_dict)
        gmo_ = fa.fit_analysis(t_range=p_dict
        ['t_range'],p_dict=p_dict,t_period=64,states=p_dict['simult_baryons_gmo'],n_states=p_dict['n_states'],prior=prior,simult=True,
        nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,
        sigma_corr_data=sigma_corr,gmo_corr_data=gmo_ratio_raw,
        model_type=model_type)
        print(gmo_)
        fit_out= gmo_.get_fit()

        out_path = 'fit_results/{0}_{1}'.format(p_dict['abbr'],model_type)
        if os.path.exists(out_path):
            pass
        else:
            os.mkdir(out_path)
        ncfg = nucleon_corr['PS'].shape[0]

        bs_list = bs.get_bs_list(Ndata=ncfg,Nbs=100)

        bscorr = ld.G_gmo_bs(file_h5=file,bs_list=bs_list,abbr=abbr,bsN=50)
        print(bscorr, "hi")


        ld.pickle_out(fit_out=fit_out,out_path=out_path,species="baryon_w_gmo")


        # if args.pdf:
        #     gmo_.make_gmo_plots(gmo_corr=gmo_ratio_raw, model_type='simult_baryons_gmo',p_dict=p_dict,
        #      fig_name='plots/{0}_{1}'.format(abbr,model_type),pdf=True)

        if args.bs:
            '''
            1) make a large number of “bootstrap copies” of the original input data and prior that differ from each other by random 
            amounts characteristic of the underlying randomness in the original data; 
            2) repeat the entire fit analysis for each bootstrap copy of the data, extracting fit results from each
            3) use the variation of the fit results from bootstrap copy to bootstrap copy to determine an approximate probability distribution (possibly non-gaussian) 
            for the fit parameters and/or functions of them: the results from each bootstrap fit are samples from that distribution.
            '''
            ncfg = nucleon_corr['PS'].shape[0]
            # ncfg_gmo = gmo_ratio_raw['PS'].shape[0]
            bs_list = bs.get_bs_list(Ndata=ncfg,Nbs=100)
            # bs_list_gmo = bs.get_bs_list(Ndata=ncfg_gmo,Nbs=100)
            # def resample_correlator(raw_corr,bs_list, n,gmo=None):
            #     if gmo:
            #         resampled_raw_corr_data = ({key : raw_corr[key][bs_list_gmo[n, :]]
            #     for key in raw_corr.keys()})
            #     else:
            #         resampled_raw_corr_data = ({key : raw_corr[key][bs_list[n, :], :]
            #     for key in raw_corr.keys()})
            #     resampled_corr_gv = resampled_raw_corr_data
            #     return resampled_corr_gv

            # def remove_nans(d):
            #     for key in d.keys():
            #         if type(d[key]) == float and isnan(d[key]):
            #             del d[key]
            #         elif type(d[key]) == gv.BufferDict():
            #             remove_nans(d[key])
            
            prelim_fit_keys = sorted(fit_out.p.keys()) 
            output = {key : [] for key in prelim_fit_keys}
            bs_N = 50
            bscorr = ld.G_gmo_bs(file_h5=file,abbr=abbr,bsN=50,bs_list=bs_list)
            print(bscorr.keys())

            for j in tqdm.tqdm(range(bs_N), desc='bootstrap'):
                gv.switch_gvar() 
                nucleon_bs  = ld.resample_correlator(nucleon_corr,bs_list=bs_list,n=j)
                xi_bs       = ld.resample_correlator(xi_corr,bs_list=bs_list,n=j)
                lam_bs      = ld.resample_correlator(lam_corr,bs_list=bs_list,n=j)
                sigma_bs    = ld.resample_correlator(sigma_corr,bs_list=bs_list,n=j)
                # gmo_bs_corr = ld.G_gmo_bs(file_h5=file,abbr=abbr,bsN=50)

                    
                # make fit with resampled correlators
                gmo_bs = fa.fit_analysis(t_range=p_dict
                ['t_range'],t_period=64,simult=True, p_dict=p_dict, states= p_dict['gmo_states'],n_states=p_dict['n_states'],prior=prior,
                nucleon_corr_data=nucleon_bs,lam_corr_data=lam_bs, xi_corr_data=xi_bs,
                sigma_corr_data=sigma_bs,delta_corr_data=None,gmo_corr_data=gmo_new,
                piplus_corr_data=None,kplus_corr_data=None,model_type=model_type)
                temp_fit = gmo_bs.get_fit()
                print(temp_fit)

                for key in prelim_fit_keys:
                    p = temp_fit.pmean[key]
                    output[key].append(p)

                gv.restore_gvar()

                # print results -- should be similar to previous results
            table = gv.dataset.avg_data(output, bstrap=True)
            print(gv.tabulate(table))


    ''' xpt routines 
    '''
    # if args.xpt:
    #     model_info = fp.model_info



    



if __name__ == "__main__":
    main()