import gvar as gv 
p_dict = {
    'abbr' : 'a09m135', #CHANGE THIS
    'part' : ['delta_pp', 'kplus', 'lambda_z', 'omega_m', 'piplus', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], #CHANGE THIS # 'proton'
    'particles' : ['proton'],#'axial_fh_num', 'vector_fh_num'],
    'fit_state' : 'xi_z',
    'gmo_states': ['sigma_p','lambda_z','proton','xi_z'], #states for gmo study
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

    't_range' : {
        'sigma' : [6, 15],
        'xi' : [6, 15],
        'proton' : [6, 15],
        'lam' : [6, 15],
        'gmo' : [6,15], #change these
    },
    'n_states' : {
        'sigma' : 2,
        'xi' :2,
        'proton':2,
        'lam':2,
        'gmo':2,
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}

priors = gv.BufferDict()
