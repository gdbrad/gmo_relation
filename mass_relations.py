# import fit_routine as fit
import numpy as np
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt
import non_analytic_functions as naf
import load_data_priors as ld
import i_o

# Set defaults for plots
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True

input_output = i_o.InputOutput()

def plot_centroid(centroid=None,t_plot_min=None,
                            t_plot_max=None, show_plot=True, show_fit=False,fig_name=None):

    


    return fig 

                            
def plot_m4(m4=None,t_plot_max=None, show_plot=True, show_fit=False,fig_name=None):
    

    return fig 



def plot_effective_mass(gmo_eff_mass=None, t_plot_min=None,
                            t_plot_max=None, show_plot=True, show_fit=False,fig_name=None):
        if t_plot_min is None:
            t_plot_min = t_min
        if t_plot_max is None:
            t_plot_max = t_max

        colors = np.array(['red', 'blue', 'green','magenta'])
        t = np.arange(t_plot_min, t_plot_max)
        effective_mass = gmo_eff_mass

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
            fit_data_gv = _generate_data_from_fit(model_type="corr", t=t)

            for j, key in enumerate(fit_data_gv.keys()):
                eff_mass_fit = get_nucleon_effective_mass(fit_data_gv, dt)[key][1:-1]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                plt.plot(t[1:-1], pm(eff_mass_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(eff_mass_fit, 1), t[1:-1], pm(eff_mass_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(eff_mass_fit, -1), pm(eff_mass_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
            plt.title("Best fit for $N_{states} = $%s" %(n_states['corr']), fontsize = 24)

        plt.xlim(t_plot_min-0.5, t_plot_max-.5)
        plt.ylim(-0.001,0.005)
        plt.legend()
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel('$GMO^{meff}$', fontsize = 24)
        fig = plt.gcf()
        plt.savefig(fig_name)
        if show_plot == True: plt.show()
        else: plt.close()

        return fig

def get_gmo_fit(fit_data=None):
    return {key : fit_data[key]
                for key in list(fit_data.keys())}

def plot_log_gmo(correlators_gv,fit_data=None, t_plot_min = None, t_plot_max = None,fig_name=None,show_fit=None,n_states=None):
    colors = np.array(['red', 'blue'])
    # print(len(colors))
    if t_plot_min == None: t_plot_min = 0
    if t_plot_max == None: t_plot_max = correlators_gv[correlators_gv.keys()[0]].shape[0] - 1

    x = range(t_plot_min, t_plot_max)
    for j, key in enumerate(sorted(correlators_gv.keys())):
        y = gv.mean(correlators_gv[key])[x]
        y_err = gv.sdev(correlators_gv[key])[x]
        
        plt.errorbar(x, y, xerr = 0.0, yerr=y_err, fmt='o', capsize=5.0,capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)
    
    if show_fit:
        # for j, key in enumerate(fit_data_gv.keys()):
        t = np.linspace(t_plot_min-2, t_plot_max+2)
        dt = (t[-1] - t[0])/(len(t) - 1)
        # fit_data_gv = fit_data

        for j, key in enumerate(fit_data.keys()):
            fit_data_ = get_gmo_fit(fit_data)[key][1:-1]

            pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
            print(fit_data_)
            print(t)
            plt.plot(t[1:-1], pm(fit_data_, 0), '--')
            #, color=colors[j%len(colors)])
            plt.plot(t[1:-1], pm(fit_data_, 1), t[1:-1], pm(fit_data_, -1))
            # color=colors[j%len(colors)])
            plt.fill_between(t[1:-1], pm(fit_data_, -1), pm(fit_data_, 1))
                            #  facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
        plt.title("Best fit for $N_{states} = $%s" %n_states, fontsize = 24)


    # Label dirac/smeared data
    plt.legend()
    plt.grid(True)
    plt.xlabel('$t$', fontsize = 24)
    plt.ylabel('$G^{GMO}(t)$', fontsize = 24)
    plt.ylim(0.99,1.01)
    fit = plt.gcf()
    plt.savefig(fig_name)
    # plt.show()
    return fit



def gmo_eff_mass(gmo_out=None,dt=None):
    '''
    calculate the "effective mass" of the gmo ratio of corr fcns as an exponential 
    '''
    if gmo_out is None:
        return None
        
    if dt is None:
        dt = 1
    return {key : 1/dt * np.log(np.exp(gmo_out[key]) / np.roll(np.exp(gmo_out[key]), -1)) for key in list(gmo_out.keys())}

class GMO(object):
    '''
    save C_2pt_{baryon}(t) as an interable object of gvars
    sea-quark masses $bm_l$ in this study are:

    TODO  
    - add M4, centroid octet mass 
    - there is a less hacky way to initialize masses below.... 
    '''
    def __init__(self,mass_dict,file=None,abbr=None):
        self.lam = mass_dict[abbr]['lam']
        self.sigma = mass_dict[abbr]['sigma']
        self.nucleon = mass_dict[abbr]['proton']
        self.xi = mass_dict[abbr]['xi']  
        self.piplus = mass_dict[abbr]['piplus']
        self.kplus = mass_dict[abbr]['kplus']
        # self.t = t
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
            if log:

                output[smr] = (np.log(
                    temp[('lambda_z', smr)]
                    * np.power(temp[('sigma_p', smr)], 1/3)
                    * np.power(temp[('proton', smr)], -2/3)
                    * np.power(temp[('xi_z', smr)], -2/3)
                ))
            else:
                output[smr] = (
                    temp[('lambda_z', smr)]
                    * np.power(temp[('sigma_p', smr)], 1/3)
                    * np.power(temp[('proton', smr)], -2/3)
                    * np.power(temp[('xi_z', smr)], -2/3)
                )

        return output

    @property 
    def gmo_violation(self):
        return self._gmo_violation

    def _gmo_violation(self):
        output = 0 
        numer = self.lam + 1/3*(self.sigma) - 2/3*self.nucleon - 2/3*self.xi 
        denom = 1/8*self.lam + 3/8*self.sigma + 1/4*self.nucleon + 1/4*self.xi
        output += numer/denom 
        return output

    @property
    def centroid(self):
        return self._centroid

    def _centroid(self):
        centroid = 1/8*self.lam + 3/8*self.sigma + 1/4*self.nucleon + 1/4*self.xi
        return centroid

        

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

class Mass_Combinations(object):


    
    def __init__(self,lam=None, sigma=None, xi=None, delta=None, sigma_st=None, xi_st=None, omega=None, p=None):
        # mev or gev? 
        self.N = 1/2 * (gv.gvar(939.565413,0.00006) + gv.gvar(938.272081,0.00006))
        self.lam = lam
        self.sigma = sigma
        self.xi = xi
        self.delta = delta
        self.sigma_st = sigma_st
        self.xi_st = xi_st
        self.Omega = omega #mev or gev
        self.N_c = 3
        self.p = p
        self.eps = (p['m_k']**2 - p['m_pi']**2) / (p['lam_chi']**2) *1e-6 # since lam_chi ~ 1 gev
        self.eps2 = self.eps**2
        self.eps3 = self.eps**3
        # isolate coefficient
        # superscript: {flavor su(3) representation, spin su(2) representation}  
        self.c = {}
        # 8 isospin multiplets of ground state baryons 
        self.c['c^{1,0}_0'] = self.m1_c
        self.c['c^{1,0}_2'] = self.m2_c   
        self.c['c^{8,0}_1'] = self.m3_c
        self.c['c^{8,0}_2']  = self.m4_c
        self.c['c^{8,0}_3'] = self.m5_c   
        self.c['c^{27,0}_2'] = self.m6_c   
        self.c['c^{27,0}_3'] = self.m7_c    
        self.c['c^{64,0}_3'] = self.m8_c

        # coeffs in jenkins paper not in 1/n_c paper:
        #    
        # self.c['c^{1,0}_1']  = self.mA      
        # self.c['c^{8,0}_35'] = self.mB
        # self.c['c^{8,0}_405'] = self.mC 
        # self.c['c^{27,0}_405'] = self.mD 

        #mass combinations given by coefficient value in table
        self.M = {}
        #su(3) singlets
        self.M['M_1'] = self.m1
        self.M['M_2'] = self.m2
        #flavor-octet
        self.M['M_3'] = self.m3
        self.M['M_4'] = self.m4
        self.M['M_5'] = self.m5
        #flavor-27 mass combs
        self.M['M_6'] = self.m6
        self.M['M_7'] = self.m7
        #supressed by 3 powers of su(3) breaking and 1/nc^2
        self.M['M_8'] = self.m8
        self.M['M_A'] = self.mA
        self.M['M_B'] = self.mB
        self.M['M_C'] = self.mC
        self.M['M_D'] = self.mD

        # scale invariant mass combination
        self.R = {}
        #su(3) singlets
        self.R['R_1'] = self.R_1
        self.R['R_2'] = self.R_2
        #flavor-octet
        self.R['R_3'] = self.R_3
        self.R['R_3_eps'] = self.R_3_eps
        self.R['R_4'] = self.R_4
        self.R['R_4_eps'] = self.R_4_eps
        self.R['R_5'] = self.R_5
        self.R['R_5_eps'] = self.R_5_eps
        #flavor-27 mass combs
        self.R['R_6'] = self.R_6
        self.R['R_6_eps2'] = self.R_6_eps2
        self.R['R_7'] = self.R_7
        self.R['R_7_eps2'] = self.R_7_eps2
        #supressed by 3 powers of su(3) breaking and 1/nc^2
        self.R['R_8'] = self.R_8
        self.R['R_8_eps3'] = self.R_8_eps3
        self.R['R_A'] = self.R_A
        self.R['R_B'] = self.R_B
        self.R['R_C'] = self.R_C
        self.R['R_D'] = self.R_D


    # calc 1/nc expansion of baryon mass operator for perturbative 
    # su3 flavor-symmetry breaking

    # M = M^1,0 + M^8,0 + M^27,0 + M^64,0
    
    @property
    def plot_baryon_mass_mpi(self):
        return self._plot_baryon_mass_mpi()
    
    def _plot_baryon_mass_mpi(self):
        mass = {}
        keys = self.M.keys()
        print(keys)
        y = np.array(self.M.values())
        print(y)
        mpi = np.array(self.p['m_pi'])
        #x = np.arange(mpi)
        print(mpi)
        fig = plt.plot(mpi,y)
        return fig
        # for k in keys:
        #     x = np.arange(keys.shape[0])
        #     print(x)
        #     mass[k] = self.M.values[x]
        #     plt.errorbar(x=x, y=[d.mean for d in mass[k]], yerr=[d.sdev for d in mass[k]], fmt='o',
        #         capsize=3, capthick=2.0, elinewidth=5.0, label=k)
        # if lim is not None:
        #     plt.xlim(lim[0], lim[1])
        #     plt.ylim(lim[2], lim[3])
        # plt.legend()
        # plt.xlabel('$t$')
        # plt.ylabel('$M_{eff}$')

        # fig = plt.gcf()
        # plt.close()
        # return fig

    #define operators


    # @property
    # def 1(self):
    #     return self._1()

    # def _1(self):
    #     output = 


    @property 
    def m1(self):
        return self._m1() * 0.001 
        #print(self.N)  

    def _m1(self):
        #output = 0
        output = 25*(2*self.N + self.lam + 3*self.sigma + 2*self.xi)
        - (4*(4*self.delta + 3*self.sigma_st + 2*self.xi_st + self.Omega))
        return output 
    @property 
    def m2(self):
        return self._m2() * 0.001

    def _m2(self):
        output = 5*(2*self.N + self.lam + 3*self.sigma + 2*self.xi)
        - (4*(4*self.delta + 3*self.sigma_st + 2*self.xi_st + self.Omega))
        return output 
    @property 
    def m3(self):
        return self._m3() * 0.0001

    def _m3(self):
        output = 5*(6*self.N + self.lam - 3*self.sigma - 4*self.xi)
        - (2*(2*self.delta - self.sigma_st - self.Omega))
        return output
    
    @property 
    def m4(self):
        return self._m4() * 0.0001
    def _m4(self):
        output = self.N + self.lam - 3*self.sigma + self.xi
        return output
    
    
    @property
    def m4_c(self):
        return self._m4_c() * 0.1

    def _m4_c(self):
        c = self.m4 / (-5* np.sqrt(3)* 1/self.N_c* self.eps)
        return c
    
    \
# scale invariant baryon mass combinations dim = mass

# TODO: PLOTS WITH ERROR BARS SUPERIMPOSED FOR /EPS #
   
    @property
    def R_4(self):
        return self._R_4()
    
    def _R_4(self):
        r = self.m4* 5*np.sqrt(3)/ 60
        return r

    @property
    def R_4_eps(self):
        return self._R_4() / self.eps

    

    

    

    

    

    

    

    


    

    

    

    

    
    
    

    

    

    



    

    


    










    
    
