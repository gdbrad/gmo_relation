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
    parser.add_argument('--fit_type',help='specify simultaneous baryon fit with or without gmo product correlator as input')
    parser.add_argument('--gmo_type',help='specify fit type for gmo product corr',required=False)
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
    nucleon_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='proton')
    lam_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='lambda_z')
    xi_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='xi_z')
    sigma_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='sigma_p')
    gmo_ratio_raw = ld.G_gmo(file,p_dict['abbr'])
    ncfg = xi_corr['PS'].shape[0]

    model_type = args.fit_type
    # prior = ld.fetch_prior(model_type,p_dict)
    # print(prior)

    if args.fit_type == 'simult_gmo_linear':
        prior_new = fp.prior
        if args.bs:
            bsN = args.bsn
            ncfg = nucleon_corr['PS'].shape[0]
            bs_list = bs.get_bs_list(Ndata=ncfg,Nbs=bsN)
            gmo_bs = ld.G_gmo_bs(file,p_dict['abbr'],bsN=bsN,bs_list=bs_list)
            gmo_direct = fa.fit_analysis(t_range=p_dict
            ['t_range'], simult=True,t_period=64,states=p_dict['simult_gmo_linear'],p_dict=p_dict, 
            n_states=p_dict['n_states'],prior=prior_new,
            nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,
            sigma_corr_data=sigma_corr,gmo_corr_data=gmo_bs,model_type=model_type)
            # print(gmo_direct)
            fit_out = gmo_direct.get_fit()
            all_fits = {}
            # print('z_gmo'+'/n',fit_out['z_gmo'])
            print('d_gmo'+'\n', fit_out['d_gmo'])
            # print('d_z_gmo'+'\n',fit_out['d_z_gmo'])
            # print('4_baryon'+'\n',fit_out['4_baryon'])
            # for observable in ['z_gmo','d_gmo','d_z_gmo','4_baryon']:
            #     print(str(observable)+'\n',fit_out[observable])
            gmo_eff_mass = gmo_direct.get_gmo_effective(gmo_ratio=gmo_ratio_raw)
            if args.pdf:
                plot1 = gmo_direct.return_best_fit_info(bs=True)
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
        else:
            gmo_direct = fa.fit_analysis(t_range=p_dict
            ['t_range'],p_dict=p_dict,t_period=64,states=p_dict['simult_baryons_gmo'],
            n_states=p_dict['n_states'],prior=prior_new,simult=True,
            nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,
            sigma_corr_data=sigma_corr,gmo_corr_data=gmo_ratio_raw,
            model_type=model_type)
            print(gmo_direct)
            fit_out= gmo_direct.get_fit()

    elif args.fit_type == 'simult_baryons':
        sim_baryons = fa.fit_analysis(t_range=p_dict
        ['t_range'],simult=True,t_period=64,states=p_dict['simult_baryons'],p_dict=p_dict, n_states=p_dict['n_states'],prior=prior,
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
    
    elif args.fit_type == 'simult_baryons_gmo':
        prior_new = fp.prior
        if args.bs:
            bsN = args.bsn
            ncfg = nucleon_corr['PS'].shape[0]
            bs_list = bs.get_bs_list(Ndata=ncfg,Nbs=bsN)
            gmo_bs = ld.G_gmo_bs(file,p_dict['abbr'],bsN=bsN,bs_list=bs_list)
            print(gmo_bs,'bs')
            gmo_direct = fa.fit_analysis(t_range=p_dict
            ['t_range'],simult=True,t_period=64,states=p_dict['simult_baryons_gmo'],p_dict=p_dict, 
            n_states=p_dict['n_states'],prior=prior_new,
            nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,
            sigma_corr_data=sigma_corr,gmo_corr_data=gmo_bs,model_type=model_type)
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
        else:
            gmo_direct = fa.fit_analysis(t_range=p_dict
            ['t_range'],p_dict=p_dict,t_period=64,states=p_dict['simult_baryons_gmo'],
            n_states=p_dict['n_states'],prior=prior_new,simult=True,
            nucleon_corr_data=nucleon_corr,lam_corr_data=lam_corr, xi_corr_data=xi_corr,
            sigma_corr_data=sigma_corr,gmo_corr_data=gmo_ratio_raw,
            model_type=model_type)
            print(gmo_direct)
            fit_out= gmo_direct.get_fit()


    elif args.fit_type == 'gmo_direct':
        if args.bs:
            bsN = args.bsn
            ncfg = nucleon_corr['PS'].shape[0]
            bs_list = bs.get_bs_list(Ndata=ncfg,Nbs=bsN)
            gmo_bs = ld.G_gmo_bs(file,p_dict['abbr'],bsN=bsN,bs_list=bs_list)
            gmo_direct = fa.fit_analysis(t_range=p_dict
            ['t_range'],simult=False,t_period=64,states=['gmo'],p_dict=p_dict, 
            n_states=p_dict['n_states'],prior=prior,
            nucleon_corr_data=None,lam_corr_data=None, xi_corr_data=None,sigma_corr_data=None,
            gmo_corr_data=gmo_bs,model_type=model_type)
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
            
        else:
            print(gmo_ratio_raw)
            prior_new = fp.prior
            gmo_direct = fa.fit_analysis(t_range=p_dict
            ['t_range'],simult=False,t_period=64,states=p_dict['gmo_direct'],p_dict=p_dict, 
            n_states=p_dict['n_states'],prior=prior_new,
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
        
    # individual correlator fits to form "naive" gmo relation
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

    
    ''' xpt routines 
    '''
    # if args.xpt:
    #     model_info = fp.model_info



    



if __name__ == "__main__":
    main()