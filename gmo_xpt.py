import numpy as np
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt
import non_analytic_functions as naf
import load_data_priors as ld
import i_o
import lsqfit


class GMO(object):
    '''
    save C_2pt_{baryon}(t) as an iterable object of gvars

    sea-quark masses $bm_l$ in this study are:

    TODO  
    - add M4, centroid octet mass 
    - there is a less hacky way to initialize masses below.... 
    '''
    def __init__(self,mass_dict=None,file=None,abbr=None,baryons=None):
        self.mass_dict = mass_dict
        self.lam = self.mass_dict[abbr]['lam']
        self.sigma = self.mass_dict[abbr]['sigma']
        self.nucleon = self.mass_dict[abbr]['proton']
        self.xi = self.mass_dict[abbr]['xi']
        if baryons:
            self.piplus = None
            self.kplus = None
        # # 
        else:
            self.piplus = self.mass_dict[abbr]['piplus']
            self.kplus = self.mass_dict[abbr]['kplus']
        # # 
        # # self.t = t
        self.file = file
        self.abbr = abbr
    @property
    def G_gmo(self):
        return self._G_gmo()

    def _G_gmo(self,log=None):
        result = {}
        # print(result)
        temp = {}
        for smr in ld.get_raw_corr(file_h5=self.file, abbr=self.abbr,particle='proton'): 
            for part in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
                temp[(part, smr)] = ld.get_raw_corr(file_h5=self.file, abbr=self.abbr,particle=part)[smr]
        temp = gv.dataset.avg_data(temp)
        # print(temp)
        output = {}
        for smr in ld.get_raw_corr(file_h5=self.file, abbr=self.abbr,particle='proton'):
            output[smr] = (
                temp[('lambda_z', smr)]
                * np.power(temp[('sigma_p', smr)], 1/3)
                * np.power(temp[('proton', smr)], -2/3)
                * np.power(temp[('xi_z', smr)], -2/3)
            )
        return output

    @property
    def gmo_rln(self):
        return self._gmo_rln
    
    def _gmo_rln(self,mev=None):
        numer = {}
        numer[self.abbr] = self.lam + 1/3*(self.sigma) - 2/3*self.nucleon - 2/3*self.xi 
        if mev:
            numer[self.abbr] = 197.3269804 * numer[self.abbr]
        return numer 

    @property 
    def gmo_violation(self):
        return self._gmo_violation
    def _gmo_violation(self):
        output = {}
        numer = self.lam + 1/3*(self.sigma) - 2/3*self.nucleon - 2/3*self.xi 
        denom = 1/8*self.lam + 3/8*self.sigma + 1/4*self.nucleon + 1/4*self.xi
        output[self.abbr] = numer/denom 
        return output
    @property 
    def m_4(self):
        return self._m4

    def _m4(self,mev=None):
        output = {}
        output[self.abbr] = self.nucleon + self.lam - 3*self.sigma + self.xi
        if mev:
            output[self.abbr] = 197.3269804 * output[self.abbr] # MeV-fm
            # output = output 
        return output

    @property
    def centroid(self):
        return self._centroid

    def _centroid(self):
        output = {}
        output[self.abbr] = 1/8*self.lam + 3/8*self.sigma + 1/4*self.nucleon + 1/4*self.xi
        return output

    

    def _get_eta_mass(self):
        '''
        no direct computation of m_eta on lattice due to issues w/ disconnected diagrams; Express in terms of meson gmo relation eg. lattice data for pion and kaon parameters.
        '''
        output = gv.gvar(
                np.power(
                4/3
                * np.power(self.kplus,2) 
                - np.power(self.piplus,2)
                ,1/2)
                )
        return output

    # def plot_mq_gmo(self):
    #     def log_gmo(self):
    #         if self.gmo is None:
    #             return ValueError("no gmo ratio provided")
    #         return np.log(self.gmo)
    #     def access_mq(self):
    #         mq = dict()
        


class GMO_xpt():
    '''
    HBxpt can be used to compute corrections to the GMO relation. 

    This should inherit all members from GMO class 
    in order to extrpaolate to the physical point. 
    Reads in PDG values
    '''
    def __init__(self,abbr,mass_dict,data,tree_level=None,loop_level=None,delta=False):
        self.data = data
        self.mass_dict = mass_dict
        self.abbr = abbr
        self.fpi = self.data[abbr]['Fpi']
        self.mpi = self.data[abbr]['mpi']
        # self.y = {'Fpi' : gv.gvar([gv.mean(g) for g in self.data[abbr]['Fpi']], [gv.sdev(g) for g in self.data[abbr]['Fpi']])}
        self.delta = delta #
        # if delta:  #TODO inclusion of delta res flag in xpt expressions

        self.params = {}
        if tree_level:
            self.params['D'] = 0.8 #axial couplings between the octet baryons
            self.params['F'] = 0.5 #axial coupling b/w octet of pseudo-Goldstone bosons,
            self.params['C'] = 1.5 #axial coupling among the decuplet of baryon resonances
        elif loop_level:
            self.params['D'] = 0.6 #axial couplings between the octet baryons
            self.params['F'] = 0.4 #axial coupling b/w octet of pseudo-Goldstone bosons,
            self.params['C'] = 1.2 #axial coupling among the decuplet of baryon resonances

        self.piplus = self.mass_dict[abbr]['piplus']
        self.kplus  = self.mass_dict[abbr]['kplus']
        self.eta    = self.mass_dict[abbr]['eta']
    @property
    def lo_gmo(self):
        return self._lo_gmo()
    def _lo_gmo(self):
        '''
        the $/mu$ independence of of baryon gmo rln is protected by the meson gmo rln, so no counterterms are required in the following lo xpt expression:
        '''

        output = {}
        # output[self.abbr] = {}

        if self.delta: 
            '''
            /delta^expt = 293 mev 
            '''
            # hbar_c = input_output.get_data_phys_point(param='hbarc')
            output = (
                1/(24*np.pi*self.fpi**2)
                * (
                    (2/3*self.params['D']**2 - 2*self.params['F']**2 )
                * (4*self.kplus**3   - 3*self.eta**3 - self.piplus**3) 
                - (self.params['C']**2)/(9 *np.pi)
                * (4*naf.fcn_F(self.kplus,self.delta) - 3*naf.fcn_F(self.eta,self.delta) - naf.fcn_F(self.piplus,self.delta)) 
                )
            )
        else:
            output = (
                1/(24*np.pi*self.y['Fpi']**2)
                * (
                    (2/3*self.params['D']**2 - 2*self.params['F']**2 )
                * (4*self.kplus**3   - 3*self.eta**3 - self.piplus**3) 
                )
            )
        return output

    
    def _lo_gmo_ratio_exp(self):
        exp_deviation = gv.gvar(0.00761,.00007)
        return self._lo_gmo()/exp_deviation

    # def _plot_deviation(self,fig_name=None):


        

    # def nlo_gmo(self):
    #     return self._nlo_gmo()

    # def _nlo_gmo(self):
    #     output = 0
    #     output += self.a1**2/(32*np.pi*self.fpi**2)* (
    #         (2/3*self.params['D']**2 - 2*self.params['F']**2 )
    #         *(4*self.data['m_k']**3 - 3*self.m_eta**3 - self.data['m_pi']**3) - self.params['C']**2/9*np.pi(4*naf.fcn_F(eps_pi=self.data['m_k']) - 3*naf.fcn_F(self.m_eta - naf.fcn_F(self.data['m_pi'])) ))
    #     return output
