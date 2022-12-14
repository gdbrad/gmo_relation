import time
import sys
import lsqfit
import os
import pandas as pd
import numpy as np
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import cmath
from gmo_fitter import fitter


class fit_ensemble(object):

    def __init__(self, t_range,t_period, prior, n_states=None, model_type = None,
                 nucleon_corr_data=None,lam_corr_data=None,
                 xi_corr_data=None,sigma_corr_data=None,
                 piplus_corr_data = None,kplus_corr_data = None
                 ):
        #All fit ensembles (manual and automatic) must have these variables
        
        #Convert correlator data into gvar dictionaries
        if nucleon_corr_data is not None:
            nucleon_corr_gv = gv.dataset.avg_data(nucleon_corr_data)
        else:
            nucleon_corr_gv = None
        if lam_corr_data is not None:
            lam_corr_gv = gv.dataset.avg_data(lam_corr_data)
        else:
            lam_corr_gv = None
        if xi_corr_data is not None:
            xi_corr_gv = gv.dataset.avg_data(xi_corr_data)
        else:
            xi_corr_gv = None
        if sigma_corr_data is not None:
            sigma_corr_gv = gv.dataset.avg_data(sigma_corr_data)
        else:
            sigma_corr_gv = None
        if piplus_corr_data is not None:
            piplus_corr_gv = gv.dataset.avg_data(piplus_corr_data)
        else:
            piplus_corr_gv = None
        if kplus_corr_data is not None:
            kplus_corr_gv = gv.dataset.avg_data(kplus_corr_data)
        else:
            kplus_corr_gv = None

        # Default to a 1 state fit
        if n_states is None:
            n_states = 1

        for data_gv in [nucleon_corr_gv]:
            if data_gv is not None:
                t_max = len(data_gv[list(data_gv.keys())[0]])


        t_start = np.min([t_range[key][0] for key in list(t_range.keys())])
        t_end = np.max([t_range[key][1] for key in list(t_range.keys())])

        max_n_states = np.max([n_states[key] for key in list(n_states.keys())])

        self.model_type = model_type
        self.nucleon_corr_gv = nucleon_corr_gv
        self.lam_corr_gv = lam_corr_gv
        self.sigma_corr_gv = sigma_corr_gv
        self.xi_corr_gv = xi_corr_gv
        self.piplus_corr_gv = piplus_corr_gv
        self.kplus_corr_gv = kplus_corr_gv

        # self.multiple_smear = None
        self.n_states = n_states
        self.prior = prior
        self.t_range = t_range
        self.t_period = t_period
        self.t_delta = 2*max_n_states
        self.t_min = int(t_start/2)
        self.t_max = int(np.min([t_end, t_end]))
        self.fits = {}
        #self.bs = None

    def get_fit(self, t_range=None, n_states=None,t_period=None):
        if t_range is None:
            t_range = self.t_range
        if n_states is None:
            n_states = self.n_states
        if t_period is None:
            t_period = self.t_period

        index = tuple((t_range[key][0], t_range[key][1], n_states[key]) for key in sorted(t_range.keys()))
        print(index)

        if index in list(self.fits.keys()):
            return self.fits[index]
        else:
            temp_fit = fitter(n_states=n_states, prior=self.prior, t_range=t_range, t_period=t_period,model_type=self.model_type,
                               nucleon_corr=self.nucleon_corr_gv,lam_corr=self.lam_corr_gv,
                               xi_corr=self.xi_corr_gv,sigma_corr=self.sigma_corr_gv, piplus_corr=self.piplus_corr_gv,
                               kplus_corr=self.kplus_corr_gv).get_fit()

            self.fits[index] = temp_fit
            return temp_fit

    # type should be either "corr, "gA", or "gV"
    def _get_models(self, model_type=None):
        
        # data_gv = [nucleon_corr , lam_corr ,sigma_corr, xi_corr]?
        if model_type is None:
            return None
        #data_gv = [nucleon_corr , lam_corr ,sigma_corr, xi_corr] 
        elif model_type == "sigma":
            nucleon_corr = None
            lam_corr = None
            sigma_corr = self.sigma_corr_gv
            xi_corr = None
            piplus_corr = None
            kplus_corr = None
        elif model_type == "xi":
            nucleon_corr = None
            lam_corr = None
            sigma_corr = None
            xi_corr = self.xi_corr_gv
            piplus_corr = None
            kplus_corr = None

        elif model_type == "lam":
            nucleon_corr = None
            lam_corr = self.lam_corr_gv
            sigma_corr = None
            xi_corr = None
            piplus_corr = None
            kplus_corr = None

        elif model_type == "proton":
            nucleon_corr = self.nucleon_corr_gv
            lam_corr = None
            sigma_corr = None
            xi_corr = None
            piplus_corr = None
            kplus_corr = None

        elif model_type == 'gmo':
            nucleon_corr = self.nucleon_corr_gv
            lam_corr = self.lam_corr_gv
            sigma_corr = self.sigma_corr_gv
            xi_corr = self.xi_corr_gv
            piplus_corr = None
            kplus_corr = None

        elif model_type == "meson":
            nucleon_corr = None
            lam_corr = None
            sigma_corr = None
            xi_corr = None
            piplus_corr = self.piplus_corr_gv
            kplus_corr = self.kplus_corr_gv


        else:
            return None 

        #print nucleon_corr_gv, axial_fh_num_gv, vector_fh_num_gv

        return fitter(n_states=self.n_states, prior=self.prior, t_range=self.t_range,t_period=self.t_period,model_type=self.model_type,
                      nucleon_corr=nucleon_corr,lam_corr=lam_corr,
                               xi_corr=xi_corr,sigma_corr=sigma_corr,
                               piplus_corr=piplus_corr,kplus_corr=kplus_corr)._make_models_simult_fit()

    def _generate_data_from_fit(self, t, t_start=None, t_end=None, model_type=None, n_states=None):
        if model_type is None:
            return None

        if t_start is None:
            t_start = self.t_range[model_type][0]
        if t_end is None:
            t_end = self.t_range[model_type][1]
        if n_states is None:
            n_states = self.n_states

        # Make
        t_range = {key : self.t_range[key] for key in list(self.t_range.keys())}
        t_range[model_type] = [t_start, t_end]

        models = self._get_models(model_type=model_type)
        fit = self.get_fit(t_range=t_range, n_states=n_states)

        # datatag[-3:] converts, eg, 'nucleon_dir' -> 'dir'
        output = {model.datatag : model.fitfcn(p=fit.p, t=t) for model in models}
        return output

    def get_nucleon_effective_mass(self, nucleon_corr_gv=None, dt=None):
        if nucleon_corr_gv is None:
            nucleon_corr_gv = self.nucleon_corr_gv

        # If still empty, return nothing
        if nucleon_corr_gv is None:
            return None

        if dt is None:
            dt = 1
        return {key : 1/dt * np.log(nucleon_corr_gv[key] / np.roll(nucleon_corr_gv[key], -1))
                for key in list(nucleon_corr_gv.keys())}

    def get_nucleon_effective_wf(self, nucleon_corr_gv=None, t=None, dt=None):
        if nucleon_corr_gv is None:
            nucleon_corr_gv = self.nucleon_corr_gv

        # If still empty, return nothing
        if nucleon_corr_gv is None:
            return None

        effective_mass = self.get_nucleon_effective_mass(nucleon_corr_gv, dt)
        if t is None:
            t = {key : np.arange(len(nucleon_corr_gv[key])) for key in list(nucleon_corr_gv.keys())}
        else:
            t = {key : t for key in list(nucleon_corr_gv.keys())}

        return {key : np.exp(effective_mass[key]*t[key]) * nucleon_corr_gv[key]
                for key in list(nucleon_corr_gv.keys())}

    def plot_effective_wf(self, nucleon_corr_gv=None, t_plot_min=None,
                           t_plot_max=None, show_plot=True, show_fit=True):
        if t_plot_min is None:
            t_plot_min = self.t_min
        if t_plot_max is None:
            t_plot_max = self.t_max

        if nucleon_corr_gv is None:
            nucleon_corr_gv = self.nucleon_corr_gv

        # If fit_ensemble doesn't have a default a nucleon correlator,
        # it's impossible to make this plot
        if nucleon_corr_gv is None:
            return None

        colors = np.array(['red', 'blue', 'green','magenta'])
        t = {}
        A_eff = {}
        for j, key in enumerate(sorted(nucleon_corr_gv.keys())):
            print(j,key)

            # plt.subplot(int(str(21)+str(j)))

            t[key] = np.arange(t_plot_min, t_plot_max)
            A_eff[key] = self.get_nucleon_effective_wf(nucleon_corr_gv)[key][t_plot_min:t_plot_max]

            pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
            lower_quantile = np.nanpercentile(gv.mean(A_eff[key]), 25)
            upper_quantile = np.nanpercentile(gv.mean(A_eff[key]), 75)
            delta_quantile = upper_quantile - lower_quantile
            plt.errorbar(x=t[key], y=gv.mean(A_eff[key]), xerr=0.0, yerr=gv.sdev(A_eff[key]),
                fmt='o', color=colors[j%len(colors)], capsize=5.0, capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)


            plt.legend()
            plt.grid(True)
            plt.ylabel('$z^{eff}$', fontsize = 24)
            plt.xlim(t_plot_min-0.5, t_plot_max-.5)
            plt.ylim(lower_quantile - 0.5*delta_quantile,
                     upper_quantile + 0.5*delta_quantile)

        if show_fit:
            t = np.linspace(t_plot_min-2, t_plot_max+2)
            dt = (t[-1] - t[0])/(len(t) - 1)
            fit_data_gv = self._generate_data_from_fit(model_type="corr", t=t)
            t = t[1:-1]
            for j, key in enumerate(sorted(fit_data_gv.keys())):
                plt.subplot(int(str(21)+str(j+1)))
                if j == 0:
                    plt.title("Best fit for $N_{states} = $%s" %(self.n_states['corr']), fontsize = 24)

                A_eff_fit = self.get_nucleon_effective_wf(fit_data_gv, t, dt)[key][1:-1]

                plt.plot(t[1:-1], pm(A_eff_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(A_eff_fit, 1), t[1:-1], pm(A_eff_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(A_eff_fit, -1), pm(A_eff_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
                plt.xlim(t_plot_min-0.5, t_plot_max-.5)

        plt.xlabel('$t$', fontsize = 24)
        fig = plt.gcf()
        plt.savefig('z_test2')
        if show_plot == True: plt.show()
        else: plt.close()

        return fig

    def plot_effective_mass(self, nucleon_corr_gv=None, t_plot_min=None, model_type=None,
                            t_plot_max=None, show_plot=True, show_fit=True,fig_name=None):
        if t_plot_min is None:
            t_plot_min = self.t_min
        if t_plot_max is None:
            t_plot_max = self.t_max

        colors = np.array(['red', 'blue', 'green','magenta'])
        t = np.arange(t_plot_min, t_plot_max)
        effective_mass = self.get_nucleon_effective_mass(nucleon_corr_gv)

        if effective_mass is None:
            return None

        y = {}
        y_err = {}
        lower_quantile = np.inf
        upper_quantile = -np.inf
        for j, key in enumerate(effective_mass.keys()):
            y[key] = gv.mean(effective_mass[key])[t]
            y_err[key] = gv.sdev(effective_mass[key])[t]
            lower_quantile = np.min([np.nanpercentile(y[key], 25), lower_quantile])
            upper_quantile = np.max([np.nanpercentile(y[key], 75), upper_quantile])

            plt.errorbar(x=t, y=y[key], xerr=0.0, yerr=y_err[key], fmt='o', capsize=5.0,
                color = colors[j%len(colors)], capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)
        delta_quantile = upper_quantile - lower_quantile
        plt.ylim(lower_quantile - 0.5*delta_quantile,
                 upper_quantile + 0.5*delta_quantile)

        if show_fit:
            t = np.linspace(t_plot_min-2, t_plot_max+2)
            dt = (t[-1] - t[0])/(len(t) - 1)
            fit_data_gv = self._generate_data_from_fit(model_type=model_type, t=t)

            for j, key in enumerate(fit_data_gv.keys()):
                eff_mass_fit = self.get_nucleon_effective_mass(fit_data_gv, dt)[key][1:-1]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                plt.plot(t[1:-1], pm(eff_mass_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(eff_mass_fit, 1), t[1:-1], pm(eff_mass_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(eff_mass_fit, -1), pm(eff_mass_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
            plt.title("Best fit for $N_{states} = $%s" %(self.n_states['gmo']), fontsize = 24)

        plt.xlim(t_plot_min-0.5, t_plot_max-.5)
        plt.legend()
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel('$M^{eff}$', fontsize = 24)
        fig = plt.gcf()
        plt.savefig(fig_name)
        if show_plot == True: plt.show()
        else: plt.close()

        return fig

    def plot_stability(self, model_type=None, t_start=None, t_end=None, t_middle=None,
                       vary_start=True, show_plot=False, n_states_array=None,fig_name=None):


        # Set axes: first for quantity of interest (eg, E0)
        ax = plt.axes([0.10,0.20,0.7,0.7])

        # Markers for identifying n_states
        markers = ["^", ">", "v", "<"]

        # Color error bars by chi^2/dof
        cmap = matplotlib.cm.get_cmap('rainbow_r')
        norm = matplotlib.colors.Normalize(vmin=0.25, vmax=1.75)

        fit_data = {}
        if n_states_array is None:
            fit_data[self.n_states[model_type]] = None
        else:
            for n_state in n_states_array:
                fit_data[n_state] = None

        if n_states_array is None:
            spacing = [0]
        else:
            spacing = (np.arange(len(n_states_array)) - (len(n_states_array)-1)/2.0)/((len(n_states_array)-1)/2.0) *0.25

        # Make fits from [t, t_end], where t is >= t_end
        if vary_start:
            if t_start is None:
                t_start = self.t_min

            if t_end is None:
                t_end = self.t_range[model_type][1]

            if t_middle is None:
                t_middle = int((t_start + 2*t_end)/3)

            plt.title("Stability plot, varying start\n Fitting [%s, %s], $N_{states} =$ %s"
                      %("$t$", t_end, sorted(fit_data.keys())), fontsize = 24)

            t = np.arange(t_start, t_middle + 1)

        # Vary end point instead
        else:
            if t_start is None:
                t_start = self.t_range[model_type][0]

            if t_end is None:
                t_end = self.t_max

            if t_middle is None:
                t_middle = int((2*t_start + t_end)/3)

            plt.title("Stability plot, varying end\n Fitting [%s, %s], $N_{states} =$ %s"
                      %(t_start, "$t$", sorted(fit_data.keys())), fontsize = 24)
            t = np.arange(t_middle, t_end + 1)

        for key in list(fit_data.keys()):
            fit_data[key] = {
                'y' : np.array([]),
                'chi2/df' : np.array([]),
                'Q' : np.array([]),
                't' : np.array([])
            }

        for n_state in list(fit_data.keys()):
            n_states_dict = self.n_states.copy()
            n_states_dict[model_type] = n_state
            for ti in t:
                t_range = {key : self.t_range[key] for key in list(self.t_range.keys())}
                if vary_start:
                    t_range[model_type] = [ti, t_end]
                    temp_fit = self.get_fit(t_range, n_states_dict)
                else:
                    t_range[model_type] = [t_start, ti]
                    temp_fit = self.get_fit(t_range, n_states_dict)
                if temp_fit is not None:
                    if model_type == 'corr':
                        fit_data[n_state]['y'] = np.append(fit_data[n_state]['y'], temp_fit.p['E0'])
                    fit_data[n_state]['chi2/df'] = np.append(fit_data[n_state]['chi2/df'], temp_fit.chi2 / temp_fit.dof)
                    fit_data[n_state]['Q'] = np.append(fit_data[n_state]['Q'], temp_fit.Q)
                    fit_data[n_state]['t'] = np.append(fit_data[n_state]['t'], ti)


        # Color map for chi/df
        cmap = matplotlib.cm.get_cmap('rainbow')
        min_max = lambda x : [np.min(x), np.max(x)]
        #minimum, maximum = min_max(fit_data['chi2/df'])
        norm = matplotlib.colors.Normalize(vmin=0.25, vmax=1.75)

        for i, n_state in enumerate(sorted(fit_data.keys())):
            for j, ti in enumerate(fit_data[n_state]['t']):
                color = cmap(norm(fit_data[n_state]['chi2/df'][j]))
                y = gv.mean(fit_data[n_state]['y'][j])
                yerr = gv.sdev(fit_data[n_state]['y'][j])

                alpha = 0.05
                if vary_start and ti == self.t_range[model_type][0]:
                    alpha=0.35
                elif (not vary_start) and ti == self.t_range[model_type][1]:
                    alpha=0.35

                plt.axvline(ti-0.5, linestyle='--', alpha=alpha)
                plt.axvline(ti+0.5, linestyle='--', alpha=alpha)

                ti = ti + spacing[i]
                plt.errorbar(ti, y, xerr = 0.0, yerr=yerr, fmt=markers[i%len(markers)], mec='k', mfc='white', ms=10.0,
                     capsize=5.0, capthick=2.0, elinewidth=5.0, alpha=0.9, ecolor=color, label=r"$N$=%s"%n_state)


        # Band for best result

        best_fit = self.get_fit()
        if model_type == 'corr':
            y_best = best_fit.p['E0']
            ylabel = r'$E_0$'

        tp = np.arange(t[0]-1, t[-1]+2)
        tlim = (tp[0], tp[-1])

        pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
        y2 = np.repeat(pm(y_best, 0), len(tp))
        y2_upper = np.repeat(pm(y_best, 1), len(tp))
        y2_lower = np.repeat(pm(y_best, -1), len(tp))

        # Ground state plot
        plt.plot(tp, y2, '--')
        plt.plot(tp, y2_upper, tp, y2_lower)
        plt.fill_between(tp, y2_lower, y2_upper, facecolor = 'yellow', alpha = 0.25)

        plt.ylabel(ylabel, fontsize=24)
        plt.xlim(tlim[0], tlim[-1])

        # Limit y-axis when comparing multiple states
        if n_states_array is not None:
            plt.ylim(pm(y_best, -5), pm(y_best, 5))

        # Get unique markers when making legend
        handles, labels = plt.gca().get_legend_handles_labels()
        temp = {}
        for j, handle in enumerate(handles):
            temp[labels[j]] = handle

        plt.legend([temp[label] for label in sorted(temp.keys())], [label for label in sorted(temp.keys())])

        ###
        # Set axes: next for Q-values
        axQ = plt.axes([0.10,0.10,0.7,0.10])

        for i, n_state in enumerate(sorted(fit_data.keys())):
            t = fit_data[n_state]['t']
            for ti in t:
                alpha = 0.05
                if vary_start and ti == self.t_range[model_type][0]:
                    alpha=0.35
                elif (not vary_start) and ti == self.t_range[model_type][1]:
                    alpha=0.35

                plt.axvline(ti-0.5, linestyle='--', alpha=alpha)
                plt.axvline(ti+0.5, linestyle='--', alpha=alpha)


            t = t + spacing[i]
            y = gv.mean(fit_data[n_state]['Q'])
            yerr = gv.sdev(fit_data[n_state]['Q'])
            color_data = fit_data[n_state]['chi2/df']

            sc = plt.scatter(t, y, vmin=0.25, vmax=1.75, marker=markers[i%len(markers)], c=color_data, cmap=cmap)

        # Set labels etc
        plt.ylabel('$Q$', fontsize=24)
        plt.xlabel('$t$', fontsize=24)
        plt.ylim(-0.05, 1.05)
        plt.xlim(tlim[0], tlim[-1])

        ###
        # Set axes: colorbar
        axC = plt.axes([0.85,0.10,0.05,0.80])

        #t = fit_data['t']
        #y = gv.mean(fit_data['g_A'])
        #yerr = gv.sdev(fit_data['g_A'])
        #color_data = fit_data['chi2/df']
        #sc = plt.scatter(t, y, vmin=0.25, vmax=1.75, c=color_data, cmap=cmap)

        colorbar = matplotlib.colorbar.ColorbarBase(axC, cmap=cmap,
                                    norm=norm, orientation='vertical')
        colorbar.set_label(r'$\chi^2_\nu$', fontsize = 24)

        fig = plt.gcf()
        plt.savefig(fig_name)
        if show_plot == True: plt.show()
        else: plt.close()

        return fig

    '''
    sensitivity of extracted spectrum and g_A on model of excited states as function of t_{sep,min}
    in C_2(t_sep) correlation function. 
    Plot:
    - prior vertical box 
    - posteriors: g_A, E0, En for 3 model types for spectrum of excitations:
    E_n = E0 + sum(l=1,n) E_l
         - harmonic-oscillator : E_n = 2m_pi
         - 1/n : E_n = 2m_pi / n
         - 1/n^2 : E_n = 2m_pi / n^2

    1. g.s. posteriors insensitive to model used
    2. excited state posteriors insensitive to model used

    * extracted spectrum highly constrained by numerical data and NOT data * 

    '''

    # def plot_sensitivity(self, model_type=None, t_start=None, t_end=None, t_middle=None,
    #                    vary_start=True, show_plot=False, n_states_array=None):
        
    #     s_mev = 197.3 / 0.08730
    #     s_gev = s_mev / 1000

    #     # set the enery levels
    #     mpi = 0.14073
    #     mN  = 0.4904

    #     models = ['SqW', 'HO', '1/n', '1/n2']
    #     models = ['HO', '1/n', '1/n2']
    #     file_m = {'SqW':'sw_result', 'HO':'ho_result', '1/n':'n_result', '1/n2':'n2_result'}
    #     fits = dict()
    #     for m in models:
    #         fits[m] = dict()
    #         for k in ['logGBF','w','Q','E0','E1','E2','E3','E4','gA','z0','pdE1', 'pdE2', 'pdE3', 'pdE4']:
    #             fits[m][k] = []
    #     for t in ['3','4','5','6','7']:
    #         for m in models:
    #             f = gv.load('ga_fit_results/spec_results_pt2/'+file_m[m]+t) # [prior, posterior, Q, logGBF]
    #             fits[m]['logGBF'].append(f[3])
    #             fits[m]['Q'].append(f[2])
    #             fits[m]['E0'].append(f[1]['E0'])
    #             for n in [1,2,3,4]:
    #                 tmp = f[1]['E0']
    #                 for l in range(1,n+1):
    #                     tmp += f[1]['dE'+str(l)]
    #                 fits[m]['E'+str(n)].append(tmp)
    #                 fits[m]['pdE'+str(n)].append(f[0]['dE'+str(n)])
    #             #print(f[0]['dE2'])
    #             fits[m]['gA'].append(f[1]['A3_00'])
    #             fits[m]['z0'].append(f[1]['z0'])

    def return_best_fit_info(self):
        plt.axis('off')
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        #plt.text(0.05, 0.05, str(fit_ensemble.get_fit(fit_ensemble.best_fit_time_range[0], fit_ensemble.best_fit_time_range[1])),
        #fontsize=14, horizontalalignment='left', verticalalignment='bottom', bbox=props)
        text = self.__str__().expandtabs()
        plt.text(0.0, 1.0, str(text),
                 fontsize=10, ha='left', va='top', family='monospace', bbox=props)

        plt.tight_layout()
        fig = plt.gcf()
        plt.close()

        return fig

    def make_plots(self, show_all=False):
        plots = np.array([])
        plots = np.append(self.return_best_fit_info(), plots)

        # Create a plot of best and stability plots
        #plots = np.append(plots, self.plot_all_fits())
        # comment out ga and gv for the hp ensembles 
        plots = np.append(plots, self.plot_effective_wf())
        plots = np.append(plots, self.plot_effective_mass())
        plots = np.append(plots, self.plot_stability(model_type='corr'))
        plots = np.append(plots, self.plot_stability(model_type='corr', vary_start=False))

        return plots

    def make_prior_from_fit(self):

        output = {}
        fit_parameters = self.get_fit().p
        for key in list(fit_parameters.keys()):
            if key == 'log(E0)' or key == 'E0':

                # Only works for protons
                # In order: proton, Roper resonance, two pions, L=1 pion excitation
                rough_energy_levels = np.array([938.0, 1440, 938+2*350,  938+2*350+110]) / 938.0
                output['E'] = gv.gvar(rough_energy_levels*gv.mean(fit_parameters['E0']),
                                      np.repeat(gv.mean(fit_parameters['E0']) * 350.0/ 938.0, 4))

            elif key == 'wf_dir':
                wf_dir = gv.gvar(0, 2*gv.mean(fit_parameters['wf_dir'][0]))
                output['wf_dir'] = np.repeat(wf_dir, 4)

            elif key == 'wf_smr':
                wf_smr = gv.gvar(gv.mean(fit_parameters['wf_smr'][0]), gv.mean(fit_parameters['wf_smr'][0]))
                output['wf_smr'] = np.repeat(wf_smr, 4)

        return output



    def __str__(self):
        output = "Fit results: \n"

        if self.nucleon_corr_gv is not None:
            output = output + "\t N_{corr} = "+str(self.n_states['corr'])+"\t"

        output = output+"\n"

        if self.nucleon_corr_gv is not None:
            output = output + "\t t_{corr} = "+str(self.t_range['corr'])

        temp_fit = self.get_fit()
        return output + str(temp_fit)
