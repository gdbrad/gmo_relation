import lsqfit
import numpy as np
import gvar as gv
import sys
import os
# local modules 
import non_analytic_functions as naf
import i_o
# import fpi_fit

class fit_routine(object):

    def __init__(self, prior, data,abbr, model_info):
         
        self.prior = prior
        self.data = data
        #self.load = i_o.InputOutput()

        self.model_info = model_info.copy()
        self.abbr = model_info['abbr']
        self.y = {datatag : self.data[abbr][datatag] for datatag in self.model_info['particles']}
        self._fit = None
        self._extrapolate = None
        self._simultaneous = False
        self._posterior = None
    def __str__(self):
        return str(self.fit)
    
    @property
    def fit(self):
        if self._fit is None:
            models = self._make_models()
            prior = self._make_prior()
            data = self.data
            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)
            self._fit = fit

        return self._fit 
    @property
    def extrapolate(self):
        if self._extrapolate is None:
            extrapolate = self.extrapolation
            self._extrapolate = extrapolate
        return self._extrapolate

    @property
    def extrapolation(self):
        extrapolation = Delta_gmo(datatag='delta_gmo',model_info=self.model_info).extrapolate_mass(observable='delta_gmo')
        return extrapolation

    # def extrapolate(self, observable=None, p=None,  data=None,c2m=1):
    #     if observable == 'sigma_pi':
    #         if p is None:
    #             p = {}
    #             p.update(self._posterior)
    #         if data is None:
    #             data = self.pp_data
    #         p.update(data)
    #         p_default = {
    #             'l3' : -1/4 * (gv.gvar('3.53(26)') + np.log(self.pp_data['eps_pi']**2))
    #             #'l4' : 
    #         }

    #         return Proton(fcn_sigma_pi(p=p, model_info=self.model_info)

    #     elif observable == 'm_p':
    #         return super().extrapolate(observable = 'eps_p', p=p, data=data)* self.pp_data


    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])

        if 'delta_gmo' in model_info['observable']:
            models = np.append(models,Delta_gmo(datatag='delta_gmo', model_info=model_info))

        return models

    def _make_prior(self, data=None):
        '''
        Only need priors for LECs/data needed in fit
        '''
        if data is None:
            data = self.data
        prior = self.prior
        new_prior = {}
        particles = []
        particles.extend(self.model_info['particles'])

        keys = []
        orders = []
        for p in particles:
            for l, value in [('xpt', self.model_info['order_chiral'])]:
            # include all orders equal to and less than desired order in expansion #
                if value == 'lo':
                    #print('hello')
                    orders = ['lo']
                elif value == 'nlo':
                    orders = ['lo', 'nlo']
                elif value == 'n2lo':
                    orders = ['lo', 'nlo','n2lo']
                else:
                    orders = []
                for o in orders:
                    keys.extend(self._get_prior_keys(particle=p, order=o, lec_type = l))
                
        for k in keys:
            new_prior[k] = prior[k]

        return new_prior

    def _get_prior_keys(self, particle = 'all', order='all',lec_type='all'):
        if particle == 'all':
            output = []
            for particle in ['piplus','kplus','eta','delta']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif order == 'all':
            output = []
            for order in ['lo', 'nlo','n2lo']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif lec_type == 'all':
            output = []
            for lec_type in ['xpt']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        else: 
            # construct dict of lec names corresponding to particle, order, lec_type #
            output = {}
            for p in ['delta_gmo']:
                output[p] = {}
                for o in ['lo', 'nlo','n2lo']:
                    output[p][o] = {}

            output['delta_gmo']['lo' ]['xpt'] = [
            
            'm_{kplus,0}','m_{eta,0}','m_{piplus,0}','m_{delta,0}']

            # if lec_type in output[particle][order]:
            #     return output[particle][order][lec_type]
            # else:
            #     return []

class Delta_gmo(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Delta_gmo, self).__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p): #data=None):
        # if data is not None:
        #     for key in data.keys():
        #         p[key] = data[key] 

        xdata = {}
        xdata['fpi'] = gv.gvar(.99,.6)
        if self.model_info['tree_level']:
            xdata['D'] = 0.8 #axial couplings between the octet baryons
            xdata['F'] = 0.5 #axial coupling b/w octet of pseudo-Goldstone bosons,
            xdata['C'] = 1.5 #axial coupling among the decuplet of baryon resonances
        elif self.model_info['loop_level']:
            xdata['D'] = 0.6 #axial couplings between the octet baryons
            xdata['F'] = 0.4 #axial coupling b/w octet of pseudo-Goldstone bosons,
            xdata['C'] = 1.2 #axial coupling among the decuplet of baryon resonances

        output = 0
       
        output += self.fitfcn_lo_xpt(p,xdata)
        # output += self.fitfcn_nlo_xpt(p,xdata) 
        # output += self.fitfcn_n2lo_xpt(p,xdata)
        return output

    def extrapolate_mass(self,observable=None,p=None, xdata=None):
        if observable == 'delta_gmo' :
            return self.fitfcn(p)
        

    def fitfcn_lo_xpt(self,p,xdata):
        '''
        the $/mu$ independence of of baryon gmo rln is protected by the meson gmo rln, so no counterterms are required in the following lo xpt expression:
        '''
        output = 0
        if self.model_info['delta']:
            output += (
                1/(24*np.pi*xdata['fpi']**2)
                * (
                    (2/3*xdata['D']**2 - 2*xdata['F']**2 )
                * (4*p['m_{kplus,0}']**3   - 3*p['m_{eta,0}']**3 - p['m_{piplus,0}']**3) 
                - (xdata['C']**2)/(9 *np.pi)
                * (4*naf.fcn_F(p['m_{kplus,0}'],p['m_{delta,0}']) - 3*naf.fcn_F(p['m_{eta,0}'],p['m_{delta,0}']) - naf.fcn_F(p['m_{piplus,0}'],p['m_{delta,0}'])) 
                )
            )
        else:
            output += (
                1/(24*np.pi*self.y['Fpi']**2)
                * (
                    (2/3*xdata['D']**2 - 2*xdata['F']**2 )
                * (4*p['m_{kplus,0}']**3   - 3*p['m_{eta,0}']**3 - p['m_{kplus,0}']**3) 
                )
            )
        return output

    # def _nlo_gmo(self):
    #     output = 0
    #     output += self.a1**2/(32*np.pi*self.fpi**2)* (
    #         (2/3*self.params['D']**2 - 2*self.params['F']**2 )
    #         *(4*self.data['m_k']**3 - 3*self.m_eta**3 - self.data['m_pi']**3) - self.params['C']**2/9*np.pi(4*naf.fcn_F(eps_pi=self.data['m_k']) - 3*naf.fcn_F(self.m_eta - naf.fcn_F(self.data['m_pi'])) ))
    #     return output

         
    # def fitfcn_n2lo_xpt(self,p,xdata):
    #     output = 0
    #     if self.model_info['order_chiral'] in ['n2lo']:
    #         if self.model_info['xpt']:
    #             if self.model_info['fit_phys_units']: 
    #                 if self.model_info['delta']:
    #                     output+= (p['g_{proton,4}'] * xdata['lam_chi']* xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'],xdata['eps_delta']))
    #                 elif self.model_info['delta'] is False:
    #                     output += p['g_{proton,4}'] * xdata['lam_chi']* xdata['eps_pi']**2

    #             elif self.model_info['fit_fpi_units']:
    #                     if self.model_info['delta']:
    #                         output+= (p['g_{proton,4}'] * xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'],xdata['eps_delta']))
    #                     elif self.model_info['delta'] is False:
    #                         output += p['g_{proton,4}'] * xdata['eps_pi']**2
    #     else:
    #         return 0
    #     return output


    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]











