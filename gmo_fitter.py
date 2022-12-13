import lsqfit
import numpy as np 
import gvar as gv

class fitter(object):

    def __init__(self, n_states,prior, t_range,
                 nucleon_corr=None,lam_corr=None,xi_corr=None,sigma_corr=None):

        self.n_states = n_states
        self.t_range = t_range
        self.prior = prior
        # self.single_smear = single_smear
        # self.data = data #TODO this should probably override below corrs
        # self.model_info = model_info.copy()
        self.lam_corr=lam_corr
        self.sigma_corr=sigma_corr
        self.nucleon_corr=nucleon_corr
        self.xi_corr=xi_corr
        self.fit = None
        self.prior = self._make_prior(prior)
        # self.fits = {}
        # self.extrapolate = None
        # self.simultaneous = False
        # self.y = {datatag : self.data['eps_'+datatag] for datatag in self.model_info['particles']}

    # @property
    # def extrapolate(self):
    #     if self._extrapolate is None:
    #         extrapolate = self.extrapolation
    #         self._extrapolate = extrapolate
    #     return self._extrapolate

    # @property
    # def extrapolation(self):
    #     extrapolation = Proton(datatag='proton',model_info=self.model_info).extrapolate_mass(observable='sigma_pi_n')
    #     return extrapolation
    
    

    def get_fit(self):
        if self.fit is not None:
            return self.fit
        else:
            return self._make_fit()

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
        # LOGIC FOR SIMULTANEOUS FITS
        # This is the convoluted way we use MultiFitter
        # Essentially: first we create a model (which is a subclass of MultiFitter)
        # Then we make a fitter using the models
        # Finally, we make the fit with our two sets of correlators

        models = self._make_models_simult_fit()
        data = self._make_data()
        # prior = self._make_prior(self.prior)
        fitter = lsqfit.MultiFitter(models=models)
        # print(fitter)
        fit = fitter.lsqfit(data=data, prior=self.prior)
        print(fit)
        self.fit = fit
        return fit

    def _make_models_simult_fit(self):
        models = np.array([])
        if self.nucleon_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.nucleon_corr.keys()):
                param_keys = {
                    'E0'      : 'E0',
                    # 'E1'      : 'E1',
                    # 'E2'      : 'E2',
                    # 'E3'      : 'E3',
                    'log(dE)' : 'log(dE)',
                    'z'      : 'z_'+sink 
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        baryon_model(datatag="nucleon_"+sink,
                        t=list(range(self.t_range['corr'][0], self.t_range['corr'][1])),
                        param_keys=param_keys, n_states=self.n_states['corr']))
        if self.lam_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.lam_corr.keys()):
                param_keys = {
                    'E0'      : 'E0',
                    # 'E1'      : 'E1',
                    # 'E2'      : 'E2',
                    # 'E3'      : 'E3',
                    'log(dE)' : 'log(dE)',
                    'z'      : 'z_'+sink 
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        baryon_model(datatag="lam_"+sink,
                        t=list(range(self.t_range['corr'][0], self.t_range['corr'][1])),
                        param_keys=param_keys, n_states=self.n_states['corr']))
        if self.xi_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.xi_corr.keys()):
                param_keys = {
                    'E0'      : 'E0',
                    # 'E1'      : 'E1',
                    # 'E2'      : 'E2',
                    # 'E3'      : 'E3',
                    'log(dE)' : 'log(dE)',
                    'z'      : 'z_'+sink 
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        baryon_model(datatag="xi_"+sink,
                        t=list(range(self.t_range['corr'][0], self.t_range['corr'][1])),
                        param_keys=param_keys, n_states=self.n_states['corr']))
        
        if self.sigma_corr is not None:
            # if self.mutliple_smear == False:
            for sink in list(self.sigma_corr.keys()):
                param_keys = {
                    'E0'      : 'E0',
                    # 'E1'      : 'E1',
                    # 'E2'      : 'E2',
                    # 'E3'      : 'E3',
                    'log(dE)' : 'log(dE)',
                    'z'      : 'z_'+sink 
                    # 'z_PS'      : 'z_PS',
                }   
                models = np.append(models,
                        baryon_model(datatag="sigma_"+sink,
                        t=list(range(self.t_range['corr'][0], self.t_range['corr'][1])),
                        param_keys=param_keys, n_states=self.n_states['corr']))

        if self.pi_corr is not None:
            for sink in list(self.pi_corr.keys()):
                param_keys = {
                    'log(E0)' : 'log(E0)',
                    'log(dE)' : 'log(dE)',
                    'z'      : 'z_'+sink,
                }
                models = np.append(models,
                           MesonModel(datatag="pi_"+sink,t=list(range(self.t_range['corr'][0], self.t_range['corr'][1]), t_period=self.t_period,
                           param_keys=param_keys, n_states=self.n_states['corr'])))

        if self.kaon_corr is not None:
            for sink in list(self.pi_corr.keys()):
                param_keys = {
                    'log(E0)' : 'log(E0)',
                    'log(dE)' : 'log(dE)',
                    'z'      : 'z_'+sink,
                }
                models = np.append(models,
                           MesonModel(datatag="pi_"+sink,t=list(range(self.t_range['corr'][0], self.t_range['corr'][1]), t_period=self.t_period,
                           param_keys=param_keys, n_states=self.n_states['corr'])))
        return models

    # data array needs to match size of t array
    def _make_data(self):
        data = {}
        if self.nucleon_corr is not None:
            for sinksrc in list(self.nucleon_corr.keys()):
                data["nucleon_"+sinksrc] = self.nucleon_corr[sinksrc][self.t_range['corr'][0]:self.t_range['corr'][1]]
        if self.lam_corr is not None:
            for sinksrc in list(self.lam_corr.keys()):
                data["lam_"+sinksrc] = self.lam_corr[sinksrc][self.t_range['corr'][0]:self.t_range['corr'][1]]
        if self.sigma_corr is not None:
            for sinksrc in list(self.sigma_corr.keys()):
                data["sigma_"+sinksrc] = self.sigma_corr[sinksrc][self.t_range['corr'][0]:self.t_range['corr'][1]]

        if self.xi_corr is not None:
            for sinksrc in list(self.xi_corr.keys()):
                data["xi_"+sinksrc] = self.xi_corr[sinksrc][self.t_range['corr'][0]:self.t_range['corr'][1]]
        return data

    def _make_prior_nested(self, prior):
        resized_prior = {}

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        gmo_list = ['sigma_p','lambda_z','proton','xi_z']
        gmo_list_ = [x for x in gmo_list]
        # print(gmo_list_)
        for corr in gmo_list:
            # print(corr)
            resized_prior[corr] = {}
            for key in list(prior[corr].keys()):
                resized_prior[corr][key] = prior[corr][key][:max_n_states]

            new_prior = resized_prior.copy()
            new_prior[corr]['E0'] = resized_prior[corr]['E'][0]
            # Don't need this entry
            new_prior.pop('E', None)

        # We force the energy to be positive by using the log-normal dist of dE
        # let log(dE) ~ eta; then dE ~ e^eta
            new_prior[corr]['log(dE)'] = gv.gvar(np.zeros(len(resized_prior[corr]['E']) - 1))
            for j in range(len(new_prior[corr]['log(dE)'])):
                #excited_state_energy = p[self.mass] + np.sum([np.exp(p[self.log_dE][k]) for k in range(j-1)], axis=0)

                # Notice that I've coded this s.t.
                # the std is determined entirely by the excited state
                # dE_mean = gv.mean(resized_prior['E'][j+1] - resized_prior['E'][j])
                # dE_std = gv.sdev(resized_prior['E'][j+1])
                temp = gv.gvar(resized_prior[corr]['E'][j+1]) - gv.gvar(resized_prior[corr]['E'][j])
                temp2 = gv.gvar(resized_prior[corr]['E'][j+1])
                temp_gvar = gv.gvar(temp.mean,temp2.sdev)
                print(temp_gvar)
                new_prior[corr]['log(dE)'][j] = np.log(temp_gvar)

        return new_prior

    def _make_prior(self,prior):
        resized_prior = {}

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        for key in list(prior.keys()):
            resized_prior[key] = prior[key][:max_n_states]

        new_prior = resized_prior.copy()
        new_prior['E0'] = resized_prior['E'][0]

        # Don't need this entry
        new_prior.pop('E', None)

        # We force the energy to be positive by using the log-normal dist of dE
        # let log(dE) ~ eta; then dE ~ e^eta
        new_prior['log(dE)'] = gv.gvar(np.zeros(len(resized_prior['E']) - 1))
        for j in range(len(new_prior['log(dE)'])):
            #excited_state_energy = p[self.mass] + np.sum([np.exp(p[self.log_dE][k]) for k in range(j-1)], axis=0)

            # Notice that I've coded this s.t.
            # the std is determined entirely by the excited state
            # dE_mean = gv.mean(resized_prior['E'][j+1] - resized_prior['E'][j])
            # dE_std = gv.sdev(resized_prior['E'][j+1])
            temp = gv.gvar(resized_prior['E'][j+1]) - gv.gvar(resized_prior['E'][j])
            temp2 = gv.gvar(resized_prior['E'][j+1])
            temp_gvar = gv.gvar(temp.mean,temp2.sdev)
            new_prior['log(dE)'][j] = np.log(temp_gvar)

        return new_prior


class baryon_model(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(baryon_model, self).__init__(datatag)
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
        z = p[self.param_keys['z']]
        # print(self.param_keys)
        E0 = p[self.param_keys['E0']]
        log_dE = p[self.param_keys['log(dE)']]
        # wf = 0
        output = z[0] * np.exp(-E0 * t)
        # print(output)
        for j in range(1, self.n_states):
            excited_state_energy = E0 + np.sum([np.exp(log_dE[k]) for k in range(j)], axis=0)
            output = output +z[j] * np.exp(-excited_state_energy * t)
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

    # def fcn_effective_mass(self, p, t=None):
    #     if t is None:
    #         t=self.t
        
    #     num = 0
    #     num += self.fitfcn()

    #     return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))

# Used for particles that obey bose-einstein statistics
class MesonModel(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, t_period, param_keys, n_states):
        super(MesonModel, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.t_period = t_period
        self.n_states = n_states

        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys


    def fitfcn(self, p, t=None):

        if t is None:
            t = self.t

        z = p[self.param_keys['z']]
        E0 = np.exp(p[self.param_keys['log(E0)']])
        dE = np.exp(p[self.param_keys['log(dE)']])

        output = z[0] * np.cosh( E0 * (t - self.t_period/2.0) )
        for j in range(1, self.n_states):
            E_j = E0 + np.sum([dE[k] for k in range(j)], axis=0)
            output = output + z[j] * np.cosh( E_j * (t - self.t_period/2.0) )

        return output


    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag][self.t]


    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.arccosh((self.fitfcn(p, t-1) + self.fitfcn(p, t+1))/(2*self.fitfcn(p, t)))


    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        return 1 / np.cosh(self.fcn_effective_mass(p, t)*(t - self.t_period/2)) * self.fitfcn(p, t)



