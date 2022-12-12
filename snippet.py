
    # Valid choices for scheme: 't0_org', 't0_imp', 'w0_org', 'w0_imp' (see hep-lat/2011.12166)
    def _get_bs_data(self, scheme=None):
        to_gvar = lambda arr : gv.gvar(arr[0], arr[1])
        hbar_c = self.load_data_phys_point('hbarc') # MeV-fm (PDG 2019 conversion constant)

        if scheme is None:
            scheme = 'w0_imp'
        if scheme not in ['t0_org', 't0_imp', 'w0_org', 'w0_imp']:
            raise ValueError('Invalid scale setting scheme')

        data = {}
        with h5py.File(self.project_path+'/data/input_data.h5', 'r') as f: 
            for ens in self.ensembles:
                data[ens] = {}
                data[ens]['units_MeV'] = hbar_c / to_gvar(f[ens]['a_fm'][scheme][:])
                data[ens]['alpha_s'] = f[ens]['alpha_s']
                data[ens]['L'] = f[ens]['L']
                data[ens]['m_pi'] = f[ens]['mpi'][:]
                data[ens]['m_k'] = f[ens]['mk'][:]
                data[ens]['lam_chi'] = 4 *np.pi *f[ens]['Fpi'][:]

                if scheme == 'w0_imp':
                    data[ens]['eps2_a'] = 1 / (2 *to_gvar(f[ens]['w0a_callat_imp']))**2
                elif scheme ==  'w0_org':
                    data[ens]['eps2_a'] = 1 / (2 *to_gvar(f[ens]['w0a_callat']))**2
                elif scheme == 't0_imp':
                    data[ens]['eps2_a'] = 1 / (4 *to_gvar(f[ens]['t0aSq_imp']))
                elif scheme == 't0_org':
                    data[ens]['eps2_a'] = 1 / (4 *to_gvar(f[ens]['t0aSq']))


        with h5py.File(self.project_path+'/data/hyperon_data.h5', 'r') as f:
            for ens in self.ensembles:
                if ens+'_hp' in list(f):
                    for obs in list(f[ens+'_hp']):
                        if obs == 'm_lam':
                            data[ens]['m_lambda'] = f[ens+'_hp'][obs][:]
                        else:
                            data[ens][obs] = f[ens+'_hp'][obs][:]
                else:
                    for obs in list(f[ens]):
                        if obs == 'm_lam':
                            data[ens]['m_lambda'] = f[ens][obs][:]
                        else:
                            data[ens][obs] = f[ens][obs][:]

        return data


    def get_data(self, scheme=None):
        bs_data = self._get_bs_data(scheme)

        gv_data = {}
        dim1_obs = ['m_delta', 'm_lambda', 'm_sigma', 'm_sigma_st', 'm_xi', 'm_xi_st', 'm_pi', 'm_k', 'lam_chi']
        for ens in self.ensembles:
            gv_data[ens] = {}
            for obs in dim1_obs:
                gv_data[ens][obs] = bs_data[ens][obs] - np.mean(bs_data[ens][obs]) + bs_data[ens][obs][0]

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 
            for obs in dim1_obs:
                gv_data[ens][obs] = gv_data[ens][obs] *bs_data[ens]['units_MeV']

            gv_data[ens]['eps2_a'] = bs_data[ens]['eps2_a']

        ensembles = list(gv_data)
        output = {}
        for param in gv_data[self.ensembles[0]]:
            output[param] = np.array([gv_data[ens][param] for ens in self.ensembles])
        return output, ensembles


    def get_data_phys_point(self, param=None):
        data_phys_point = {
            'eps2_a' : gv.gvar(0),
            'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : gv.gvar(197.3269804, 0), # MeV-fm

            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)'),
            'm_pi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'm_k' : gv.gvar('494.2(3)'), # '495.6479(92)'

            'm_lambda' : gv.gvar(1115.683, 0.006),
            'm_sigma' : np.mean([gv.gvar(g) for g in ['1189.37(07)', '1192.642(24)', '1197.449(30)']]),
            'm_sigma_st' : np.mean([gv.gvar(g) for g in ['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)']]),
            'm_xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
            'm_xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
        }
        if param is not None:
            return data_phys_point[param]
        return data_phys_point