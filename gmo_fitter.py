import lsqfit
import numpy as np 
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt

class fitter(object):

    def __init__(self, n_states,prior, t_period,t_range,states,
                 nucleon_corr=None,p_dict=None,lam_corr=None,
                 xi_corr=None,sigma_corr=None,
                 gmo_ratio_corr=None,
                 model_type=None,simult=None,gmo_type = None):

        self.n_states = n_states
        self.t_period = t_period
        self.t_range = t_range
        self.prior = prior
        self.p_dict = p_dict
        self.lam_corr=lam_corr
        self.sigma_corr=sigma_corr
        self.nucleon_corr=nucleon_corr
        self.xi_corr=xi_corr
        self.gmo_ratio_corr = gmo_ratio_corr
        self.fit = None
        self.model_type = model_type
        self.simult = simult
        self.gmo_type = gmo_type
        self.states = states
        self.prior = self._make_prior(prior)
        effective_mass = {}
        self.effective_mass = effective_mass
        # self.fits = {}
        # self.extrapolate = None
        # self.y = {datatag : self.data['eps_'+datatag] for datatag in self.model_info['particles']}

    def return_best_fit_info(self):
        plt.axis('off')
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        #plt.text(0.05, 0.05, str(fit_ensemble.get_fit(fit_ensemble.best_fit_time_range[0], fit_ensemble.best_fit_time_range[1])),
        #fontsize=14, horizontalalignment='left', verticalalignment='bottom', bbox=props)
        text = self.__str__().expandtabs()
        plt.text(0.0, 1.0, str(text),
                 fontsize=16, ha='left', va='top', family='monospace', bbox=props)

        plt.tight_layout()
        fig = plt.gcf()
        plt.close()

        return fig

    def __str__(self):
        output = "Model Type:" + str(self.model_type) 
        output = output+"\n"

        output = output + "\t N_{corr} = "+str(self.n_states[self.model_type])+"\t"
        output = output+"\n"
        output += "Fit results: \n"

        output += str(self.get_fit())
        return output


    def _generate_data_from_fit(self, t,fit=None, t_start=None, t_end=None, model_type=None, n_states=None):
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

        # models = self._get_models(model_type=model_type)
        if fit is None:

            fit = self.get_fit(t_range=t_range, n_states=n_states)

        # datatag[-3:] converts, eg, 'nucleon_dir' -> 'dir'
        output = {model.datatag : model.fitfcn(p=fit.p, t=t) for model in models}
        return output


    def get_fit(self):
        if self.fit is not None:
            return self.fit
        else:
            return self._make_fit()

    def fcn_effective_mass(self, t, t_start=None, t_end=None, n_states=None):
        if t_start is None:
            t_start = self.t_range[self.model_type][0]
        if t_end is None:
            t_end = self.t_range[self.model_type][1]
        if n_states is None: n_states = self.n_states

        p = self.get_fit().p
        output = {}
        for model in self._make_models_simult_fit():
            snk = model.datatag
            output[snk] = model.fcn_effective_mass(p, t)
        return output

    def plot_effective_mass(self, tmin=None, tmax=None, ylim=None, show_fit=True,show_plot=True,fig_name=None):
        if tmin is None: tmin = 1
        if tmax is None: tmax = self.t_period - 1

        fig = self._plot_quantity(
            quantity=self.effective_mass, 
            ylabel=r'$m_\mathrm{eff}$', 
            tmin=tmin, tmax=tmax, ylim=ylim) 

        if show_fit:
            ax = plt.gca()

            colors = ['rebeccapurple', 'mediumseagreen']
            t = np.linspace(tmin, tmax)
            effective_mass_fit = self.fcn_effective_mass(t=t)
            for j, snk in enumerate(sorted(effective_mass_fit)):
                color = colors[j%len(colors)]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                ax.plot(t, pm(effective_mass_fit[snk], 0), '--', color=color)
                ax.plot(t, pm(effective_mass_fit[snk], 1), 
                            t, pm(effective_mass_fit[snk], -1), color=color)
                ax.fill_between(t, pm(effective_mass_fit[snk], -1), pm(effective_mass_fit[snk], 1), facecolor=color, alpha = 0.10, rasterized=True)

        fig = plt.gcf()
        if show_plot:
            plt.show()
        plt.close()
        return fig

    def _plot_quantity(self, quantity,
            tmin, tmax, ylabel=None, ylim=None):

        fig, ax = plt.subplots()
        
        colors = ['rebeccapurple', 'mediumseagreen']
        for j, snk in enumerate(sorted(quantity)):
            x = np.arange(tmin, tmax)
            y = gv.mean(quantity[snk])[x]
            y_err = gv.sdev(quantity[snk])[x]

            ax.errorbar(x, y, xerr = 0.0, yerr=y_err, fmt='o', capsize=5.0,
                        color=colors[j%len(quantity)], capthick=2.0, alpha=0.6, elinewidth=5.0, label=snk)

        # Label dirac/smeared data
        #plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.legend(loc=3, bbox_to_anchor=(0,1), ncol=len(quantity))
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel(ylabel, fontsize = 24)

        if ylim is not None:
            plt.ylim(ylim)
        fig = plt.gcf()
        return fig

    def get_energies(self):

        # Don't rerun the fit if it's already been made
        if self.fit is not None:
            temp_fit = self.fit
        else:
            temp_fit = self.get_fit()

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        output = gv.gvar(np.zeros(max_n_states))
        output[0] = temp_fit.p['E0']
        for k in range(1, max_n_states):
            output[k] = output[0] + np.sum([(temp_fit.p['dE'][j]) for j in range(k)], axis=0)
        return output

    def _make_fit(self):
        # Essentially: first we create a model (which is a subclass of MultiFitter)
        # Then we make a fitter using the models
        # Finally, we make the fit with our two sets of correlators

        models = self._make_models_simult_fit()
        data = self._make_data()
        fitter = lsqfit.MultiFitter(models=models)
        fit = fitter.lsqfit(data=data, prior=self.prior)
        self.fit = fit
        return fit

    def _make_models_simult_fit(self):
        models = np.array([])

        if self.nucleon_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.nucleon_corr.keys()):
                param_keys = {
                    'proton_E0'      : 'proton_E0',
                    'proton_log(dE)' : 'proton_log(dE)',
                    'proton_z'      : 'proton_z_'+sink 
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        proton_model(datatag="nucleon_"+sink,
                        t=list(range(self.t_range['proton'][0], self.t_range['proton'][1])),
                        param_keys=param_keys, n_states=self.n_states['proton']))
        if self.lam_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.lam_corr.keys()):
                param_keys = {
                    'lam_E0'      : 'lam_E0',
                    'lam_log(dE)' : 'lam_log(dE)',
                    'lam_z'      : 'lam_z_'+sink 
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        lam_model(datatag="lam_"+sink,
                        t=list(range(self.t_range['lam'][0], self.t_range['lam'][1])),
                        param_keys=param_keys, n_states=self.n_states['lam']))
        if self.xi_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.xi_corr.keys()):
                param_keys = {
                    'xi_E0'      : 'xi_E0',
                    'xi_log(dE)' : 'xi_log(dE)',
                    'xi_z'      : 'xi_z_'+sink 
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        xi_model(datatag="xi_"+sink,
                        t=list(range(self.t_range['xi'][0], self.t_range['xi'][1])),
                        param_keys=param_keys, n_states=self.n_states['xi']))
        
        if self.sigma_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.sigma_corr.keys()):
                param_keys = {
                    'sigma_E0'      : 'sigma_E0',
                    'sigma_log(dE)' : 'sigma_log(dE)',
                    'sigma_z'      : 'sigma_z_'+sink 
                }   
                models = np.append(models,
                        sigma_model(datatag="sigma_"+sink,
                        t=list(range(self.t_range['sigma'][0], self.t_range['sigma'][1])),
                        param_keys=param_keys, n_states=self.n_states['sigma']))


        if self.model_type == 'gmo_direct':
            for sink in list(self.gmo_ratio_corr.keys()):
                param_keys = {
                    'gmo_E0'    : 'gmo_E0',
                    'z'         : 'gmo_z_'+sink,
                    'log(dE)' : 'gmo_log(dE)',
                                    }
                models = np.append(models,
                        GMO_direct(datatag="gmo_ratio_"+sink,
                        t=list(range(self.t_range['gmo_ratio'][0], self.t_range['gmo_ratio'][1])),
                        param_keys=param_keys, n_states=self.n_states['gmo_ratio']))

        if self.model_type == 'simult_baryons_gmo':
            for sink in list(self.gmo_ratio_corr.keys()):

                param_keys = {
                    'sigma_E0'  : 'sigma_E0',
                    'xi_E0'     : 'xi_E0',
                    'lam_E0'    : 'lam_E0',
                    'proton_E0' : 'proton_E0',
                    'p_log(dE)' : 'proton_log(dE)',
                    'x_log(dE)' : 'xi_log(dE)',
                    's_log(dE)' : 'sigma_log(dE)',
                    'l_log(dE)' : 'lam_log(dE)',
                    'proton_z'  : 'proton_z_'+sink,
                    'lam_z'     : 'lam_z_'+sink,
                    'xi_z'      : 'xi_z_'+sink,
                    'sigma_z'   : 'sigma_z_'+sink 
                                    }
                models = np.append(models,
                        gmo_model(datatag="gmo_ratio_"+sink,
                        t=list(range(self.t_range['gmo_ratio'][0], self.t_range['gmo_ratio'][1])),
                        param_keys=param_keys, n_states=self.n_states['gmo_ratio']))

        if self.model_type == 'simult_gmo_linear':
            for sink in list(self.gmo_ratio_corr.keys()):

                param_keys = {
                    'gmo_E0'    : 'gmo_E0',
                    'z_gmo'     : 'z_gmo',
                    'gmo_log(dE)': 'gmo_log(dE)',
                    'gmo_z'      : 'gmo_z_'+sink,
                    'sigma_E0'  : 'sigma_E0',
                    'xi_E0'     : 'xi_E0',
                    'lam_E0'    : 'lam_E0',
                    'proton_E0' : 'proton_E0',
                    'p_log(dE)' : 'proton_log(dE)',
                    'x_log(dE)' : 'xi_log(dE)',
                    's_log(dE)' : 'sigma_log(dE)',
                    'l_log(dE)' : 'lam_log(dE)',
                    'proton_z'  : 'proton_z_'+sink,
                    'lam_z'     : 'lam_z_'+sink,
                    'xi_z'      : 'xi_z_'+sink,
                    'sigma_z'   : 'sigma_z_'+sink 
                                    }
                models = np.append(models,
                        gmo_linear_model(datatag="gmo_ratio_"+sink,gmo_type=self.gmo_type,
                        t=list(range(self.t_range['gmo_ratio'][0], self.t_range['gmo_ratio'][1])),
                        param_keys=param_keys, n_states=self.n_states['gmo_ratio']))

        return models

    # data array needs to match size of t array
    def _make_data(self):
        data = {}

        if self.gmo_ratio_corr is not None:
            for sinksrc in list(self.gmo_ratio_corr.keys()):
                data["gmo_ratio_"+sinksrc] = self.gmo_ratio_corr[sinksrc][self.t_range['proton'][0]:self.t_range['proton'][1]]
        if self.nucleon_corr is not None:
            for sinksrc in list(self.nucleon_corr.keys()):
                data["nucleon_"+sinksrc] = self.nucleon_corr[sinksrc][self.t_range['proton'][0]:self.t_range['proton'][1]]
        if self.lam_corr is not None:
            for sinksrc in list(self.lam_corr.keys()):
                data["lam_"+sinksrc] = self.lam_corr[sinksrc][self.t_range['lam'][0]:self.t_range['lam'][1]]
        if self.sigma_corr is not None:
            for sinksrc in list(self.sigma_corr.keys()):
                data["sigma_"+sinksrc] = self.sigma_corr[sinksrc][self.t_range['sigma'][0]:self.t_range['sigma'][1]]
        if self.xi_corr is not None:
            for sinksrc in list(self.xi_corr.keys()):
                data["xi_"+sinksrc] = self.xi_corr[sinksrc][self.t_range['xi'][0]:self.t_range['xi'][1]]
        return data

    def _make_prior(self,prior):
        resized_prior = {}

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        for key in list(prior.keys()):
            resized_prior[key] = prior[key][:max_n_states]

        new_prior = resized_prior.copy()
        if self.simult:
            for corr in ['sigma','lam','proton','xi','gmo']:
                new_prior[corr+'_E0'] = resized_prior[corr+'_E'][0]
                new_prior.pop(corr+'_E', None)
                new_prior[corr+'_log(dE)'] = gv.gvar(np.zeros(len(resized_prior[corr+'_E']) - 1))
                for j in range(len(new_prior[corr+'_log(dE)'])):
                    temp = gv.gvar(resized_prior[corr+'_E'][j+1]) - gv.gvar(resized_prior[corr+'_E'][j])
                    temp2 = gv.gvar(resized_prior[corr+'_E'][j+1])
                    temp_gvar = gv.gvar(temp.mean,temp2.sdev)
                    new_prior[corr+'_log(dE)'][j] = np.log(temp_gvar)
        else:
            for corr in self.states:
                new_prior[corr+'_E0'] = resized_prior[corr+'_E'][0]
                new_prior.pop(corr+'_E', None)

        # We force the energy to be positive by using the log-normal dist of dE
        # let log(dE) ~ eta; then dE ~ e^eta
                new_prior[corr+'_log(dE)'] = gv.gvar(np.zeros(len(resized_prior[corr+'_E']) - 1))
                for j in range(len(new_prior[corr+'_log(dE)'])):
                    temp = gv.gvar(resized_prior[corr+'_E'][j+1]) - gv.gvar(resized_prior[corr+'_E'][j])
                    temp2 = gv.gvar(resized_prior[corr+'_E'][j+1])
                    temp_gvar = gv.gvar(temp.mean,temp2.sdev)
                    new_prior[corr+'_log(dE)'][j] = np.log(temp_gvar)

        return new_prior

class GMO_direct(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(GMO_direct, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t
        z = p[self.param_keys['z']]
        E0 = p[self.param_keys['gmo_E0']]
        log_dE = p[self.param_keys['log(dE)']]
        output = z[0] * np.exp(-E0 * t)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]

class gmo_linear_model(lsqfit.MultiFitterModel):
    '''
    Product of the four baryon correlation functions modeled as a decaying exponential. Asymptotes to the GMO Relation ~0. 
    TODO is the normalization coefficient the single octet relation ?? 
    We treat the linear combination of the 4 baryons as a single fit parameter, then fit to 3rd? order in the taylor expansion with $/delta B$ as the overlap factor for the product correlator 
    '''
    def __init__(self, gmo_type,datatag, t, param_keys, n_states):
        super(gmo_linear_model, self).__init__(datatag)
        # variables for fit
        self.gmo_type = gmo_type
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t
        
        gmo_E0 = p[self.param_keys['gmo_E0']]
        z_gmo =  p[self.param_keys['z_gmo']]
        gmo_log_dE = p[self.param_keys['gmo_log(dE)']]
        proton_E0 = p[self.param_keys['proton_E0']]
        sigma_E0 = p[self.param_keys['sigma_E0']]
        xi_E0 =  p[self.param_keys['xi_E0']]
        lam_E0  = p[self.param_keys['lam_E0']]
        proton_log_dE = p[self.param_keys['p_log(dE)']]
        sigma_log_dE = p[self.param_keys['s_log(dE)']]
        xi_log_dE = p[self.param_keys['x_log(dE)']]
        lam_log_dE = p[self.param_keys['l_log(dE)']]
        z_p = p[self.param_keys['proton_z']]
        z_s = p[self.param_keys['sigma_z']]
        z_x = p[self.param_keys['xi_z']]
        z_l = p[self.param_keys['lam_z']]
        output_gmo = 0

        if self.gmo_type == 'd_gmo':
            '''
            ratio of overlap factors determined entirely by baryons
            fitting d_gmo 
            '''
            A_0 = z_l[0]*np.power(z_s[0],1/3)*np.power(z_p[0],-2/3) * np.power(z_x[0],-2/3)
            output_gmo += A_0 * np.exp(-gmo_E0 * t)

        elif self.gmo_type == 'z_gmo':
            '''
            overlap factor Z_GMO as ind. fit param
            '''
            A_0 = p[self.param_keys['gmo_z']]
            delta_gmo = (lam_E0 + 1/3*sigma_E0 - 2/3*proton_E0 - 2/3*xi_E0)
            output_gmo += A_0[0] * np.exp(-delta_gmo * t)

        elif self.gmo_type == 'd_z_gmo':
            '''
            fitting both z_gmo and d_gmo
            '''
            A_0 = p[self.param_keys['gmo_z']]
            output_gmo += A_0[0] * np.exp(-gmo_E0 * t)

        elif self.gmo_type == '4_baryon':
            '''
            fit params determined entirely by baryons
            '''
            A_0 = z_l[0]*np.power(z_s[0],1/3)*np.power(z_p[0],-2/3) * np.power(z_x[0],-2/3)
            delta_gmo = (lam_E0 + 1/3*sigma_E0 - 2/3*proton_E0 - 2/3*xi_E0)
            output_gmo += A_0 * np.exp(-delta_gmo * t)

        for j in range(1, self.n_states):
            p_esc = proton_E0 + np.sum([np.exp(proton_log_dE[k]) for k in range(j)], axis=0)
            output_p = z_p[j] * np.exp(-p_esc*t)
        for j in range(1, self.n_states):
            s_esc    = sigma_E0+ np.sum([np.exp(sigma_log_dE[k]) for k in range(j)], axis=0)
            output_s = z_s[j] * np.exp(-s_esc*t)
        for j in range(1, self.n_states):
            x_esc    = xi_E0+ np.sum([np.exp(xi_log_dE[k]) for k in range(j)], axis=0)
            output_x = z_x[j] * np.exp(-x_esc*t)
        for j in range(1, self.n_states):
            l_esc    =  lam_E0+ np.sum([np.exp(lam_log_dE[k]) for k in range(j)], axis=0)
            output_l = z_l[j] * np.exp(-l_esc*t)
            
        output_gmo =(
            output_gmo*( 
                (1+output_l) 
            * (np.power(1+output_s,1/3))
            * (np.power(1+output_p,-2/3) 
            * (np.power(1+output_x,-2/3))
            )))
        # for j in range(1, self.n_states):
        #     d_gmo_esc =  gmo_E0+ np.sum([np.exp(gmo_log_dE[k]) for k in range(j)], axis=0)
        # output_gmo +=  (B_l[1] + 1/3*B_s[1] - 2/3*B_p[1] - 2/3*B_x[1] + delta_B[1]) * np.exp(-d_gmo_esc * t)
        return output_gmo

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]


class gmo_model(lsqfit.MultiFitterModel):
    '''
    Product of the four baryon correlation functions modeled as a decaying exponential. Asymptotes to the GMO Relation ~0. 
    TODO is the normalization coefficient the single octet relation ?? 

    We treat the linear combination of the 4 baryons as a single fit parameter, then fit to 3rd? order in the taylor expansion with $/delta B$ as the overlap factor for the product correlator 
    '''
    def __init__(self, datatag, t, param_keys, n_states):
        super(gmo_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t
        
        proton_E0 = p[self.param_keys['proton_E0']]
        sigma_E0 = p[self.param_keys['sigma_E0']]
        xi_E0 =  p[self.param_keys['xi_E0']]
        lam_E0  = p[self.param_keys['lam_E0']]
        proton_log_dE = p[self.param_keys['p_log(dE)']]
        sigma_log_dE = p[self.param_keys['s_log(dE)']]
        xi_log_dE = p[self.param_keys['x_log(dE)']]
        lam_log_dE = p[self.param_keys['l_log(dE)']]
        z_p = p[self.param_keys['proton_z']]
        z_s = p[self.param_keys['sigma_z']]
        z_x = p[self.param_keys['xi_z']]
        z_l = p[self.param_keys['lam_z']]
        
        output_p = z_p[0] * np.exp(-proton_E0 * t)
        for j in range(1, self.n_states):
            p_esc = proton_E0 + np.sum([np.exp(proton_log_dE[k]) for k in range(j)], axis=0)
            output_p = output_p + z_p[j] * np.exp(-p_esc*t)
        output_s = z_s[0] * np.exp(-sigma_E0 * t)
        for j in range(1, self.n_states):
            s_esc    = sigma_E0 + np.sum([np.exp(sigma_log_dE[k]) for k in range(j)], axis=0)
            output_s = output_s + z_s[j] * np.exp(-s_esc*t)
        output_x = z_x[0] * np.exp(-xi_E0 * t)
        for j in range(1, self.n_states):
            x_esc    = xi_E0 + np.sum([np.exp(xi_log_dE[k]) for k in range(j)], axis=0)
            output_x = output_x + z_x[j] * np.exp(-x_esc*t)
        output_l = z_l[0] * np.exp(-lam_E0 * t)
        for j in range(1, self.n_states):
            l_esc    =  lam_E0 + np.sum([np.exp(lam_log_dE[k]) for k in range(j)], axis=0)
            output_l = output_l + z_l[j] * np.exp(-l_esc*t)
            
        output = output_l * np.power(output_s,1/3)*np.power(output_p,-2/3) * np.power(output_x,-2/3)

        return output

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]
class proton_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(proton_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t
        z = p[self.param_keys['proton_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['proton_E0']]
        log_dE = p[self.param_keys['proton_log(dE)']]
        output = z[0] * np.exp(-E0 * t)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        t = np.array(t)
        
        return np.exp(self.fcn_effective_mass(p, t)*t) * self.fitfcn(p, t)


    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]

class lam_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(lam_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        # z_PS = p[self.param_keys['z_PS']]
        # z_SS = p[self.param_keys['z_SS']]
        z = p[self.param_keys['lam_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['lam_E0']]
        log_dE = p[self.param_keys['lam_log(dE)']]
        # wf = 0
        output = z[0] * np.exp(-E0 * t)
        # print(output)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]

class xi_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(xi_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        # z_PS = p[self.param_keys['z_PS']]
        # z_SS = p[self.param_keys['z_SS']]
        z = p[self.param_keys['xi_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['xi_E0']]
        log_dE = p[self.param_keys['xi_log(dE)']]
        # wf = 0
        output = z[0] * np.exp(-E0 * t)
        # print(output)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]

    

class sigma_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(sigma_model, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states
        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys

    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        # z_PS = p[self.param_keys['z_PS']]
        # z_SS = p[self.param_keys['z_SS']]
        z = p[self.param_keys['sigma_z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['sigma_E0']]
        log_dE = p[self.param_keys['sigma_log(dE)']]
        # wf = 0
        output = z[0] * np.exp(-E0 * t)
        # print(output)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
        return output

    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior

    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag]