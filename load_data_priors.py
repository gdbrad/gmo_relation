import gvar as gv 
import h5py as h5 
import numpy as np 

def get_corr(file_h5,abbr):
    data = {}
    with h5.File(file_h5,"r") as f:
        path = "/"+ abbr 
        particles = f[path].keys()
        for part in particles:
            cfgs = np.array(f[path+"/" + part][()])
            # print(cfgs)
            data[part] = cfgs.real # .real for the _hp ensembles 
    return data

def fetch_prior(p_dict,states):

    prior_nucl = {}
    newlist = [x for x in states]
    for i,x in enumerate(newlist):
        path = os.path.normpath("./priors/{0}/{1}/prior_nucl.csv".format(p_dict['abbr'],x))
        df = pd.read_csv(path, index_col=0).to_dict()
        for key in list(df.keys()):
            length = int(np.sqrt(len(list(df[key].values()))))
            prior_nucl[key] = list(df[key].values())[:length]
        prior = {}
        prior[x] = {}
        prior[x] = gv.gvar(prior_nucl)
        print(prior)
        # prior = {**prior_nucl}
        # prior_nucl = gv.BufferDict()
        # prior[x] = prior_nucl
    return prior
