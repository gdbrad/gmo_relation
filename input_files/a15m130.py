import gvar as gv 
p_dict = {
    'abbr' : 'a15m130', #CHANGE THIS
    'part' : ['delta_pp', 'kplus', 'lambda_z', 'omega_m', 'piplus', 'proton', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'particles' : ['proton'],
    'meson_states' : ['piplus','kplus'],
    'gmo_states': ['sigma_p','lambda_z','proton','xi_z'], #states for gmo study
    'gmo_states_all' : ['delta','sigma_p','lambda_z','proton','xi_z','piplus','kplus'],
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma' : [6, 16],
        'xi' : [4, 15],
        'proton' : [6, 15],
        'delta' : [4,15],
        'lam' : [4, 15],
        'gmo' : [6,15], 
        'pi' : [4,25],
        'kplus': [4,25],
    },
    'n_states' : {
        'sigma' : 2,
        'xi' :3,
        'delta':3,
        'proton':2,
        'lam':2,
        'gmo':2,
        'pi' : 2,
        'kplus': 2,
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}

# TODO put prior routines in here, filename save options 
priors = gv.BufferDict()
