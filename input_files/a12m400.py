import gvar as gv 
import numpy as np
p_dict = {
    'abbr' : 'a12m400', #CHANGE THIS
    'part' : ['delta_pp', 'kplus', 'lambda_z', 'omega_m', 'piplus', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'gmo_direct' : ['gmo'],
    'meson_states' : ['piplus','kplus'],
    'simult_baryons': ['sigma_p','lambda_z','proton','xi_z'],
    'simult_baryons_gmo': ['sigma_p','lambda_z','proton','xi_z','gmo'],
    'simult_gmo_linear': ['gmo'], #states for gmo study
     #states for gmo study
    'srcs'     :['S'],
    'snks'     :['SS','PS'],
    'bs_seed' : 'a12m400',

   't_range' : {
        'sigma' : [10,20],
        'xi' :  [10,20],
        'proton' :   [10,20],
        'delta' : [10,20],
        'lam' : [10,20],
        'gmo' : [2,10], 
        'pi' : [5,30],
        'kplus': [8,28],
	    'gmo_ratio':[10,20],
	    'gmo_direct':[10,20],
        'simult_baryons':   [10,20],
        'simult_baryons_gmo':[10,20],
        'simult_gmo_linear':[10,20],

    },
    'n_states' : {
        'sigma' : 2,
        'xi' :2,
        'delta':2,
        'proton':2,
        'lam':2,
        'pi' : 2,
        'gmo':2,
        'kplus': 2,
        'mesons':2,
	    'gmo_ratio':2,
        'gmo_direct':2,
        'simult_baryons':2,
        'simult_baryons_gmo':2,
        'simult_gmo_linear':2
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}
prior = gv.BufferDict()
prior = {
    'sigma_E': np.array(['0.83(22)', '0.9(3.2)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'sigma_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'lam_E': np.array(['0.7(2.2)', '0.9(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'proton_E': np.array(['0.61(22)', '0.9(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'proton_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'xi_E': np.array(['0.8(2.2)', '1.28(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_z_SS': np.array(['0.000012(12)', '0.000012(12)', '0.000012(12)', '0.000012(12)'],dtype=object),
    'gmo_E': np.array(['0.001(10)', '0.004(15)'], dtype=object),
    'gmo_z_PS': np.array(['0.6(60)', '0.6(60)'], dtype=object),
    'gmo_z_SS': np.array(['0.6(60)', '0.6(60)'], dtype=object),
    'delta_B': np.array(['0.7(7.5)', '0.7(7.5)'], dtype=object),}

'''
$\delta_{GMO}$ xpt extrapolation model and prior information
'''
model_info = {}
model_info['particles'] = ['piplus','kplus','eta']
model_info['order_chiral'] = 'lo'
model_info['tree_level'] = True
model_info['loop_level'] = False
model_info['delta'] = True
model_info['abbr'] = ['a12m180L']
model_info['observable'] = ['delta_gmo'] #'centroid', 'octet'

# prior = {}
# prior['m_{kplus,0}'] = gv.gvar(0.35,.1)
# prior['m_{eta,0}'] = gv.gvar(.3,.2)
# prior['m_{piplus,0}'] = gv.gvar(.25,.1)
# prior['m_{delta,0}'] = gv.gvar(2,1)

# TODO put prior routines in here, filename save options 
