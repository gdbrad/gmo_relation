import gvar as gv 
p_dict = {
    'abbr' : 'a12m180L', #CHANGE THIS
    'part' : ['delta_pp', 'kplus', 'lambda_z', 'omega_m', 'piplus', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], #CHANGE THIS # 'proton'
    'particles' : ['proton'],#'axial_fh_num', 'vector_fh_num'],
    'fit_state' : 'xi_z',
    'gmo_states': ['sigma_p','lambda_z','proton','xi_z'], #states for gmo study
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

    't_range' : {
        'corr' : [6, 13], #change these
    },
    'n_states' : {
        'corr' : 2,
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}

priors = gv.BufferDict()
