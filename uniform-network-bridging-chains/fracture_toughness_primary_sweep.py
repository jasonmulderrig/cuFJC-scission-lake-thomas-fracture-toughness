"""The single-chain fracture toughness primary sweep characterization
module for cuFJCs that undergo scission
"""

# import external modules
from __future__ import division
import numpy as np
from math import floor, log10
from scipy import constants
import matplotlib.pyplot as plt
from mpi4py import MPI
from cufjc_scission import (
    RateIndependentScissioncuFJC,
    RateDependentScissioncuFJC
)
import time

# import internal modules
from .characterizer import Characterizer
from .utility import (
    latex_formatting_figure,
    save_current_figure,
    save_pickle_object,
    load_pickle_object
)


class FractureToughnessPrimarySweepCharacterizer(Characterizer):
    """The characterization class assessing fracture toughness for
    composite uFJCs that undergo scission. It inherits all attributes
    and methods from the ``Characterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``FractureToughnessPrimarySweepCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``Characterizer`` class.
        """
        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        Characterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        # nu = 1 -> nu = 10000, only 151 unique nu values exist here
        nu_list = np.unique(np.rint(np.logspace(0, 4, 195)))
        nu_num = len(nu_list)
        nu_list_mpi_split = np.array_split(nu_list, self.comm_size)
        nu_num_list_mpi_split = [
            len(nu_list_mpi_split[proc_indx])
            for proc_indx in range(self.comm_size)
        ]
        p.characterizer.nu_list = nu_list
        p.characterizer.nu_num = nu_num
        p.characterizer.nu_list_mpi_split = nu_list_mpi_split
        p.characterizer.nu_num_list_mpi_split = nu_num_list_mpi_split

        sample_nu_list = [10, 100, 1000]
        sample_nu_num = len(sample_nu_list)
        sample_nu_label_list = [
            r'$\nu='+str(sample_nu_list[sample_nu_indx])+'$'
            for sample_nu_indx in range(sample_nu_num)
        ]
        sample_nu_color_list = ['blue', 'green', 'red']

        p.characterizer.sample_nu_list = sample_nu_list
        p.characterizer.sample_nu_num = sample_nu_num
        p.characterizer.sample_nu_label_list = sample_nu_label_list
        p.characterizer.sample_nu_color_list = sample_nu_color_list

        p.characterizer.nu_val = 25

        p.characterizer.lmbda_c_eq_min = 0.001
        p.characterizer.lmbda_c_eq_num_steps = 2501

        p.characterizer.xi_c_num_steps = 100001
        p.characterizer.lmbda_c_eq_num_steps = 100001

        p.characterizer.zeta_nu_char_base_val = 100
        p.characterizer.kappa_nu_base_val = 1000

        segment_extensibility_list = [100, 500, 1000, 5000, 10000]

        zeta_nu_char_list = np.copy(segment_extensibility_list)
        # Reverse the order of the zeta_nu_char list
        zeta_nu_char_list = zeta_nu_char_list[::-1]
        zeta_nu_char_num = len(zeta_nu_char_list)
        zeta_nu_char_label_list = [
            r'$\zeta_{\nu}^{char}='+str(zeta_nu_char_list[zeta_nu_char_indx])+'$'
            for zeta_nu_char_indx in range(zeta_nu_char_num)
        ]
        zeta_nu_char_color_list = ['orange', 'blue', 'green', 'red', 'purple']

        p.characterizer.zeta_nu_char_list = zeta_nu_char_list
        p.characterizer.zeta_nu_char_num = zeta_nu_char_num
        p.characterizer.zeta_nu_char_label_list = zeta_nu_char_label_list
        p.characterizer.zeta_nu_char_color_list = zeta_nu_char_color_list

        kappa_nu_list = np.copy(segment_extensibility_list)
        kappa_nu_num = len(kappa_nu_list)
        kappa_nu_label_list = [
            r'$\kappa_{\nu}='+str(kappa_nu_list[kappa_nu_indx])+'$'
            for kappa_nu_indx in range(kappa_nu_num)
        ]
        kappa_nu_color_list = ['orange', 'blue', 'green', 'red', 'purple']

        p.characterizer.kappa_nu_list = kappa_nu_list
        p.characterizer.kappa_nu_num = kappa_nu_num
        p.characterizer.kappa_nu_label_list = kappa_nu_label_list
        p.characterizer.kappa_nu_color_list = kappa_nu_color_list

        nondim_rate_prmtr_list = np.logspace(-21, 3, 177)

        check_xi_c_dot_list = np.copy(nondim_rate_prmtr_list)
        check_xi_c_dot_num = len(check_xi_c_dot_list)
        check_xi_c_dot_list_mpi_split = np.array_split(
            check_xi_c_dot_list, self.comm_size)
        check_xi_c_dot_num_list_mpi_split = [
            len(check_xi_c_dot_list_mpi_split[proc_indx])
            for proc_indx in range(self.comm_size)
        ]
        p.characterizer.check_xi_c_dot_list = check_xi_c_dot_list
        p.characterizer.check_xi_c_dot_num = check_xi_c_dot_num
        p.characterizer.check_xi_c_dot_list_mpi_split = (
            check_xi_c_dot_list_mpi_split
        )
        p.characterizer.check_xi_c_dot_num_list_mpi_split = (
            check_xi_c_dot_num_list_mpi_split
        )

        tilde_xi_c_dot_list = np.copy(nondim_rate_prmtr_list)
        tilde_xi_c_dot_num = len(tilde_xi_c_dot_list)
        tilde_xi_c_dot_list_mpi_split = np.array_split(tilde_xi_c_dot_list, self.comm_size)
        tilde_xi_c_dot_num_list_mpi_split = [
            len(tilde_xi_c_dot_list_mpi_split[proc_indx])
            for proc_indx in range(self.comm_size)
        ]
        p.characterizer.tilde_xi_c_dot_list = tilde_xi_c_dot_list
        p.characterizer.tilde_xi_c_dot_num = tilde_xi_c_dot_num
        p.characterizer.tilde_xi_c_dot_list_mpi_split = tilde_xi_c_dot_list_mpi_split
        p.characterizer.tilde_xi_c_dot_num_list_mpi_split = (
            tilde_xi_c_dot_num_list_mpi_split
        )

        sample_tilde_xi_c_dot_list = [1e-20, 1e-10, 1]
        sample_tilde_xi_c_dot_num = len(sample_tilde_xi_c_dot_list)
        sample_tilde_xi_c_dot_exponent_list = [
            int(floor(log10(abs(sample_tilde_xi_c_dot_list[sample_tilde_xi_c_dot_indx]))))
            for sample_tilde_xi_c_dot_indx in range(sample_tilde_xi_c_dot_num)
        ]
        sample_tilde_xi_c_dot_label_list = [
            r'$\tilde{\dot{\xi}}_c='+'10^{0:d}'.format(sample_tilde_xi_c_dot_exponent_list[sample_tilde_xi_c_dot_indx])
            for sample_tilde_xi_c_dot_indx in range(sample_tilde_xi_c_dot_num)
        ]
        sample_tilde_xi_c_dot_color_list = ['blue', 'green', 'red']

        p.characterizer.sample_tilde_xi_c_dot_list = sample_tilde_xi_c_dot_list
        p.characterizer.sample_tilde_xi_c_dot_num = sample_tilde_xi_c_dot_num
        p.characterizer.sample_tilde_xi_c_dot_label_list = (
            sample_tilde_xi_c_dot_label_list
        )
        p.characterizer.sample_tilde_xi_c_dot_color_list = (
            sample_tilde_xi_c_dot_color_list
        )

    def prefix(self):
        """Set characterization prefix"""
        return "fracture_toughness_primary_sweep"
    
    def characterization(self):
        """Define characterization routine"""

        T = 298 # absolute room temperature, K
        k_B = constants.value(u'Boltzmann constant') # J/K
        h = constants.value(u'Planck constant') # J/Hz
        hbar = h / (2*np.pi) # J*sec
        beta = 1. / (k_B*T) # 1/J
        omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec

        cp = self.parameters.characterizer
        
        nu_list_mpi_scatter = self.comm.scatter(cp.nu_list_mpi_split, root=0)
        check_xi_c_dot_list_mpi_scatter = self.comm.scatter(
            cp.check_xi_c_dot_list_mpi_split, root=0)
        tilde_xi_c_dot_list_mpi_scatter = self.comm.scatter(
            cp.tilde_xi_c_dot_list_mpi_split, root=0)

        # Rate-independent calculations
        
        if self.comm_rank == 0:
            print("Rate-independent calculations")

            base_A_nu__sample_nu_list = []

            for nu_val in cp.sample_nu_list:
                single_chain = (
                    RateIndependentScissioncuFJC(nu=nu_val,
                            zeta_nu_char=cp.zeta_nu_char_base_val,
                            kappa_nu=cp.kappa_nu_base_val)
                )
                A_nu_val = single_chain.A_nu
                base_A_nu__sample_nu_list.append(A_nu_val)
            
            save_pickle_object(
                self.savedir,
                base_A_nu__sample_nu_list, "base_A_nu__sample_nu_list")
        
            chain_mech_resp_zeta_nu_char_single_chain_list = [
                RateIndependentScissioncuFJC(
                    nu=cp.nu_val,
                    zeta_nu_char=cp.zeta_nu_char_list[zeta_nu_char_indx],
                    kappa_nu=cp.kappa_nu_base_val)
                for zeta_nu_char_indx in range(cp.zeta_nu_char_num)
            ]

            chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk = []
            chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk = []

            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                single_chain = (
                    chain_mech_resp_zeta_nu_char_single_chain_list[zeta_nu_char_indx]
                )

                lmbda_c_eq_steps = (
                    np.linspace(
                        cp.lmbda_c_eq_min, single_chain.lmbda_c_eq_crit,
                        cp.lmbda_c_eq_num_steps)
                )

                lmbda_c_eq = []
                xi_c = []

                for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_steps):
                    lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
                    lmbda_nu_val = single_chain.lmbda_nu_func(lmbda_c_eq_val)
                    xi_c_val = single_chain.xi_c_func(
                        lmbda_nu_val, lmbda_c_eq_val)

                    lmbda_c_eq.append(lmbda_c_eq_val)
                    xi_c.append(xi_c_val)
                
                chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk.append(
                    lmbda_c_eq)
                chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk.append(
                    xi_c)
            
            chain_mech_resp_kappa_nu_single_chain_list = [
                RateIndependentScissioncuFJC(
                    nu=cp.nu_val,
                    zeta_nu_char=cp.zeta_nu_char_base_val,
                    kappa_nu=cp.kappa_nu_list[kappa_nu_indx])
                for kappa_nu_indx in range(cp.kappa_nu_num)
            ]

            chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk = []
            chain_mech_resp_kappa_nu_xi_c___single_chain_chunk = []

            for kappa_nu_indx in range(cp.kappa_nu_num):
                single_chain = (
                    chain_mech_resp_kappa_nu_single_chain_list[kappa_nu_indx]
                )

                lmbda_c_eq_steps = (
                    np.linspace(
                        cp.lmbda_c_eq_min, single_chain.lmbda_c_eq_crit,
                        cp.lmbda_c_eq_num_steps)
                )

                lmbda_c_eq = []
                xi_c = []

                for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_steps):
                    lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
                    lmbda_nu_val = single_chain.lmbda_nu_func(lmbda_c_eq_val)
                    xi_c_val = single_chain.xi_c_func(
                        lmbda_nu_val, lmbda_c_eq_val)

                    lmbda_c_eq.append(lmbda_c_eq_val)
                    xi_c.append(xi_c_val)
                
                chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk.append(
                    lmbda_c_eq)
                chain_mech_resp_kappa_nu_xi_c___single_chain_chunk.append(
                    xi_c)

            save_pickle_object(
                self.savedir,
                chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk,
                "chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk")
            save_pickle_object(
                self.savedir,
                chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk,
                "chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk")
            save_pickle_object(
                self.savedir,
                chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk,
                "chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk")
            save_pickle_object(
                self.savedir,
                chain_mech_resp_kappa_nu_xi_c___single_chain_chunk,
                "chain_mech_resp_kappa_nu_xi_c___single_chain_chunk")
        
        zeta_nu_char_A_nu___nu_chunk_list_mpi_scatter = []
        kappa_nu_A_nu___nu_chunk_list_mpi_scatter = []
        base_A_nu___nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            zeta_nu_char_A_nu = []
            kappa_nu_A_nu = []

            for zeta_nu_char_val in cp.zeta_nu_char_list:
                single_chain = (
                    RateIndependentScissioncuFJC(nu=nu_val,
                          zeta_nu_char=zeta_nu_char_val,
                          kappa_nu=cp.kappa_nu_base_val)
                )
                A_nu_val = single_chain.A_nu
                zeta_nu_char_A_nu.append(A_nu_val)
            
            for kappa_nu_val in cp.kappa_nu_list:
                single_chain = (
                    RateIndependentScissioncuFJC(nu=nu_val,
                          zeta_nu_char=cp.zeta_nu_char_base_val,
                          kappa_nu=kappa_nu_val)
                )
                A_nu_val = single_chain.A_nu
                kappa_nu_A_nu.append(A_nu_val)
            
            zeta_nu_char_A_nu___nu_chunk_list_mpi_scatter.append(
                zeta_nu_char_A_nu
            )
            kappa_nu_A_nu___nu_chunk_list_mpi_scatter.append(
                kappa_nu_A_nu
            )

            single_chain = (
                RateIndependentScissioncuFJC(nu=nu_val,
                        zeta_nu_char=cp.zeta_nu_char_base_val,
                        kappa_nu=cp.kappa_nu_base_val)
            )
            A_nu_val = single_chain.A_nu
            base_A_nu___nu_chunk_list_mpi_scatter.append(A_nu_val)
        
        zeta_nu_char_A_nu___nu_chunk_list_mpi_split = self.comm.gather(
            zeta_nu_char_A_nu___nu_chunk_list_mpi_scatter, root=0)
        kappa_nu_A_nu___nu_chunk_list_mpi_split = self.comm.gather(
            kappa_nu_A_nu___nu_chunk_list_mpi_scatter, root=0)
        base_A_nu___nu_chunk_list_mpi_split = self.comm.gather(
            base_A_nu___nu_chunk_list_mpi_scatter, root=0)
        
        self.comm.Barrier()


        if self.comm_rank == 0:
            print("Post-processing rate-independent calculations")
            
            zeta_nu_char_A_nu___nu_chunk_list = []
            kappa_nu_A_nu___nu_chunk_list = []
            base_A_nu___nu_chunk_list = []

            for proc_indx in range(self.comm_size):
                for nu_chunk_indx in range(cp.nu_num_list_mpi_split[proc_indx]):
                    zeta_nu_char_A_nu___nu_chunk_val = (
                        zeta_nu_char_A_nu___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    kappa_nu_A_nu___nu_chunk_val = (
                        kappa_nu_A_nu___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    base_A_nu___nu_chunk_val = (
                        base_A_nu___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    zeta_nu_char_A_nu___nu_chunk_list.append(
                        zeta_nu_char_A_nu___nu_chunk_val)
                    kappa_nu_A_nu___nu_chunk_list.append(
                        kappa_nu_A_nu___nu_chunk_val)
                    base_A_nu___nu_chunk_list.append(base_A_nu___nu_chunk_val)
            
            save_pickle_object(
                self.savedir, zeta_nu_char_A_nu___nu_chunk_list,
                "zeta_nu_char_A_nu___nu_chunk_list")
            save_pickle_object(
                self.savedir, kappa_nu_A_nu___nu_chunk_list,
                "kappa_nu_A_nu___nu_chunk_list")
            save_pickle_object(
                self.savedir, base_A_nu___nu_chunk_list,
                "base_A_nu___nu_chunk_list")
        
        self.comm.Barrier()


        # Rate-dependent calculations

        if self.comm_rank == 0:
            print("Rate-dependent calculations")
            print("Rate-dependent check_xi_c_dot versus zeta_nu_char primary sweep")

        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []

        for check_xi_c_dot_val in check_xi_c_dot_list_mpi_scatter:
            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit = []
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit = []

            for zeta_nu_char_val in cp.zeta_nu_char_list:
                rate_dependent_single_chain = (
                    RateDependentScissioncuFJC(nu=cp.nu_val,
                                               zeta_nu_char=zeta_nu_char_val,
                                               kappa_nu=cp.kappa_nu_base_val,
                                               omega_0=omega_0)
                )
                xi_c_dot_val = check_xi_c_dot_val * omega_0 * cp.nu_val
                xi_c_steps = np.linspace(
                    0, rate_dependent_single_chain.xi_c_crit, cp.xi_c_num_steps)
                t_steps = xi_c_steps / xi_c_dot_val # 1/(1/sec) = sec
            
                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for xi_c_indx in range(cp.xi_c_num_steps):
                    t_val = t_steps[xi_c_indx]
                    xi_c_val = xi_c_steps[xi_c_indx]
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(
                            xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(
                            lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if xi_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[xi_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[xi_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char_val
                )

                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
            
            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit)
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit)
        
        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )

        self.comm.Barrier()


        if self.comm_rank == 0:
            print("Rate-dependent check_xi_c_dot versus kappa_nu primary sweep")

        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []

        for check_xi_c_dot_val in check_xi_c_dot_list_mpi_scatter:
            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit = []
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit = []

            for kappa_nu_val in cp.kappa_nu_list:
                rate_dependent_single_chain = (
                    RateDependentScissioncuFJC(nu=cp.nu_val,
                                               zeta_nu_char=cp.zeta_nu_char_base_val,
                                               kappa_nu=kappa_nu_val,
                                               omega_0=omega_0)
                )
                xi_c_dot_val = check_xi_c_dot_val * omega_0 * cp.nu_val
                xi_c_steps = np.linspace(
                    0, rate_dependent_single_chain.xi_c_crit, cp.xi_c_num_steps)
                t_steps = xi_c_steps / xi_c_dot_val # 1/(1/sec) = sec
            
                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for xi_c_indx in range(cp.xi_c_num_steps):
                    t_val = t_steps[xi_c_indx]
                    xi_c_val = xi_c_steps[xi_c_indx]
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(
                            xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(
                            lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if xi_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[xi_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[xi_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / cp.zeta_nu_char_base_val
                )

                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
            
            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit)
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit)
        
        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )

        self.comm.Barrier()


        if self.comm_rank == 0:
            print("Rate-dependent nu primary sweep with sampled tilde_xi_c_dot")
        
        rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_dependent_epsilon_cnu_diss_hat_crit = []
            rate_dependent_overline_epsilon_cnu_diss_hat_crit = []

            for tilde_xi_c_dot_val in cp.sample_tilde_xi_c_dot_list:
                rate_dependent_single_chain = (
                    RateDependentScissioncuFJC(nu=nu_val,
                                               zeta_nu_char=cp.zeta_nu_char_base_val,
                                               kappa_nu=cp.kappa_nu_base_val,
                                               omega_0=omega_0)
                )
                xi_c_dot_val = tilde_xi_c_dot_val * omega_0
                xi_c_steps = np.linspace(
                    0, rate_dependent_single_chain.xi_c_crit, cp.xi_c_num_steps)
                t_steps = xi_c_steps / xi_c_dot_val # 1/(1/sec) = sec
            
                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for xi_c_indx in range(cp.xi_c_num_steps):
                    t_val = t_steps[xi_c_indx]
                    xi_c_val = xi_c_steps[xi_c_indx]
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(
                            xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(
                            lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if xi_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[xi_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[xi_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / cp.zeta_nu_char_base_val
                )

                rate_dependent_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
            
            rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_epsilon_cnu_diss_hat_crit)
            rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_overline_epsilon_cnu_diss_hat_crit)
        
        rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )

        self.comm.Barrier()


        if self.comm_rank == 0:
            print("Rate-dependent tilde_xi_c_dot primary sweep with sampled nu")
        
        rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_scatter = []

        for tilde_xi_c_dot_val in tilde_xi_c_dot_list_mpi_scatter:
            rate_dependent_epsilon_cnu_diss_hat_crit = []
            rate_dependent_overline_epsilon_cnu_diss_hat_crit = []

            for nu_val in cp.sample_nu_list:
                rate_dependent_single_chain = (
                    RateDependentScissioncuFJC(nu=nu_val,
                                               zeta_nu_char=cp.zeta_nu_char_base_val,
                                               kappa_nu=cp.kappa_nu_base_val,
                                               omega_0=omega_0)
                )
                xi_c_dot_val = tilde_xi_c_dot_val * omega_0
                xi_c_steps = np.linspace(
                    0, rate_dependent_single_chain.xi_c_crit, cp.xi_c_num_steps)
                t_steps = xi_c_steps / xi_c_dot_val # 1/(1/sec) = sec
            
                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for xi_c_indx in range(cp.xi_c_num_steps):
                    t_val = t_steps[xi_c_indx]
                    xi_c_val = xi_c_steps[xi_c_indx]
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(
                            xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(
                            lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if xi_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[xi_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[xi_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / cp.zeta_nu_char_base_val
                )

                rate_dependent_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
            
            rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_epsilon_cnu_diss_hat_crit)
            rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_overline_epsilon_cnu_diss_hat_crit)
        
        rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )
        
        
        self.comm.Barrier()


        if self.comm_rank == 0:
            print("Post-processing rate-dependent calculations")

            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []

            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []

            for proc_indx in range(self.comm_size):
                for check_xi_c_dot_chunk_indx in range(cp.check_xi_c_dot_num_list_mpi_split[proc_indx]):
                    rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )

                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )

                    rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)

                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)
            
            rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list = []
            rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list = []

            for proc_indx in range(self.comm_size):
                for nu_chunk_indx in range(cp.nu_num_list_mpi_split[proc_indx]):
                    rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_val)
                    rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val)
            
            rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list = []
            rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list = []

            for proc_indx in range(self.comm_size):
                for tilde_xi_c_dot_chunk_indx in range(cp.tilde_xi_c_dot_num_list_mpi_split[proc_indx]):
                    rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_val = (
                        rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_val = (
                        rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    )

                    rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list.append(
                        rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_val)
                    rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list.append(
                        rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_val)
            
            save_pickle_object(
                self.savedir,
                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list,
                "rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list,
                "rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list,
                "rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list,
                "rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                "rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                "rate_dependent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list,
                "rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list,
                "rate_dependent_overline_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list")

    def finalization(self):
        """Define finalization analysis"""

        ### I still need to make the plots showing rate-independent
        ### results versus sampled results with tilde_xi_c_dot

        if self.comm_rank == 0:
            print("Fracture toughness sweep finalization")

        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing
        
        if self.comm_rank == 0:
            print("Plotting")

            base_A_nu__sample_nu_list = (
                load_pickle_object( self.savedir, "base_A_nu__sample_nu_list")
            )

            chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk = (
                load_pickle_object(
                    self.savedir,
                    "chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk")
            )
            chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk = (
                load_pickle_object(
                    self.savedir,
                    "chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk")
            )
            chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk = (
                load_pickle_object(
                    self.savedir,
                    "chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk")
            )
            chain_mech_resp_kappa_nu_xi_c___single_chain_chunk = (
                load_pickle_object(
                    self.savedir,
                    "chain_mech_resp_kappa_nu_xi_c___single_chain_chunk")
            )

            zeta_nu_char_A_nu___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "zeta_nu_char_A_nu___nu_chunk_list")
            )
            kappa_nu_A_nu___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "kappa_nu_A_nu___nu_chunk_list")
            )
            base_A_nu___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "base_A_nu___nu_chunk_list")
            )

            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            )

            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            )

            rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            )

            rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_epsilon_cnu_diss_hat_crit___tilde_xi_c_dot_chunk_list")
            )

            # plot results
            latex_formatting_figure(ppp)

            fig = plt.figure()
            for kappa_nu_indx in range(cp.kappa_nu_num):
                lmbda_c_eq_list = (
                    chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk[kappa_nu_indx]
                )
                xi_c_list = (
                    chain_mech_resp_kappa_nu_xi_c___single_chain_chunk[kappa_nu_indx]
                )
                plt.semilogy(lmbda_c_eq_list, xi_c_list, linestyle='-',
                    color=cp.kappa_nu_color_list[kappa_nu_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.kappa_nu_label_list[kappa_nu_indx])
                plt.plot(lmbda_c_eq_list[-1], xi_c_list[-1], marker='x',
                    color=cp.kappa_nu_color_list[kappa_nu_indx],
                    alpha=1, linewidth=2.5)
            plt.legend(loc='best', fontsize=10)
            plt.xlim([-0.05, 2.05])
            plt.xticks(fontsize=20)
            plt.ylim([1e-2, 1e4])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_c$', 30,
                "semilogy-kappa_nu-xi_c-vs-lmbda_c_eq")
            
            fig = plt.figure()
            for kappa_nu_indx in range(cp.kappa_nu_num):
                lmbda_c_eq_list = (
                    chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk[kappa_nu_indx]
                )
                xi_c_list = (
                    chain_mech_resp_kappa_nu_xi_c___single_chain_chunk[kappa_nu_indx]
                )
                plt.plot(lmbda_c_eq_list, xi_c_list, linestyle='-',
                    color=cp.kappa_nu_color_list[kappa_nu_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.kappa_nu_label_list[kappa_nu_indx])
                plt.plot(lmbda_c_eq_list[-1], xi_c_list[-1], marker='x',
                    color=cp.kappa_nu_color_list[kappa_nu_indx],
                    alpha=1, linewidth=2.5)
            plt.legend(loc='best', fontsize=10)
            plt.xlim([-0.05, 2.05])
            plt.xticks(fontsize=20)
            plt.ylim([0, 125])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_c$', 30,
                "kappa_nu-xi_c-vs-lmbda_c_eq")
            
            fig = plt.figure()
            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                lmbda_c_eq_list = (
                    chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk[zeta_nu_char_indx]
                )
                xi_c_list = (
                    chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk[zeta_nu_char_indx]
                )
                plt.semilogy(lmbda_c_eq_list, xi_c_list, linestyle='-',
                    color=cp.zeta_nu_char_color_list[zeta_nu_char_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.zeta_nu_char_label_list[zeta_nu_char_indx])
                plt.plot(lmbda_c_eq_list[-1], xi_c_list[-1], marker='x',
                    color=cp.zeta_nu_char_color_list[zeta_nu_char_indx],
                    alpha=1, linewidth=2.5)
            plt.legend(loc='best', fontsize=10)
            plt.xlim([-0.05, 4.25])
            plt.xticks(fontsize=20)
            plt.ylim([1e-2, 1e4])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_c$', 30,
                "semilogy-zeta_nu_char-xi_c-vs-lmbda_c_eq")
            
            fig = plt.figure()
            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                lmbda_c_eq_list = (
                    chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk[zeta_nu_char_indx]
                )
                xi_c_list = (
                    chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk[zeta_nu_char_indx]
                )
                plt.plot(lmbda_c_eq_list, xi_c_list, linestyle='-',
                    color=cp.zeta_nu_char_color_list[zeta_nu_char_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.zeta_nu_char_label_list[zeta_nu_char_indx])
                plt.plot(lmbda_c_eq_list[-1], xi_c_list[-1], marker='x',
                    color=cp.zeta_nu_char_color_list[zeta_nu_char_indx],
                    alpha=1, linewidth=2.5)
            plt.legend(loc='best', fontsize=10)
            plt.xlim([-0.05, 4.25])
            plt.xticks(fontsize=20)
            plt.ylim([0, 350])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_c$', 30,
                "zeta_nu_char-xi_c-vs-lmbda_c_eq")
            
            kappa_nu_A_nu___nu_chunk_arr = (
                np.asarray(kappa_nu_A_nu___nu_chunk_list)
            )
            A_nu___kappa_nu_chunk_arr = (
                np.transpose(kappa_nu_A_nu___nu_chunk_arr)
            )
            
            fig = plt.figure()
            for kappa_nu_indx in range(cp.kappa_nu_num):
                A_nu_list = A_nu___kappa_nu_chunk_arr[kappa_nu_indx]
                plt.semilogx(
                    cp.nu_list, A_nu_list,
                    linestyle='-',
                    color=cp.kappa_nu_color_list[kappa_nu_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.kappa_nu_label_list[kappa_nu_indx])
            plt.legend(loc='best', fontsize=10)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$\mathcal{A}_{\nu}$', 30,
                "kappa_nu-A_nu-vs-nu")
            
            zeta_nu_char_A_nu___nu_chunk_arr = (
                np.asarray(zeta_nu_char_A_nu___nu_chunk_list)
            )
            A_nu___zeta_nu_char_chunk_arr = (
                np.transpose(zeta_nu_char_A_nu___nu_chunk_arr)
            )
            
            fig = plt.figure()
            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                A_nu_list = A_nu___zeta_nu_char_chunk_arr[zeta_nu_char_indx]
                plt.semilogx(
                    cp.nu_list, A_nu_list,
                    linestyle='-',
                    color=cp.zeta_nu_char_color_list[zeta_nu_char_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.zeta_nu_char_label_list[zeta_nu_char_indx])
            plt.legend(loc='best', fontsize=10)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$\mathcal{A}_{\nu}$', 30,
                "zeta_nu_char-A_nu-vs-nu")

            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list)
            )
            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr = (
                np.transpose(
                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )
            
            fig = plt.figure()
            for kappa_nu_indx in range(cp.kappa_nu_num):
                rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit_list = (
                    rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr[kappa_nu_indx]
                )
                plt.semilogx(
                    cp.check_xi_c_dot_list,
                    rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-',
                    color=cp.kappa_nu_color_list[kappa_nu_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.kappa_nu_label_list[kappa_nu_indx])
            plt.legend(loc='best', fontsize=10)
            plt.xlim([1e-20, 1e0])
            plt.xticks([1e-20, 1e-10, 1e0], fontsize=20)
            plt.ylim([-1, 51])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\check{\dot{\xi}}_c$', 30,
                r'$(\hat{\varepsilon}_{c\nu}^{diss})^{crit}$', 30,
                "kappa_nu-epsilon_cnu_diss_hat_crit-vs-check_xi_c_dot")

            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list)
            )
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr = (
                np.transpose(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )
            
            fig = plt.figure()
            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit_list = (
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr[zeta_nu_char_indx]
                )
                plt.semilogx(
                    cp.check_xi_c_dot_list,
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-',
                    color=cp.zeta_nu_char_color_list[zeta_nu_char_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.zeta_nu_char_label_list[zeta_nu_char_indx])
            plt.legend(loc='best', fontsize=10)
            plt.xlim([1e-20, 1e0])
            plt.xticks([1e-20, 1e-10, 1e0], fontsize=20)
            plt.ylim([-0.01, 0.51])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\check{\dot{\xi}}_c$', 30,
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$', 30,
                "zeta_nu_char-overline_epsilon_cnu_diss_hat_crit-vs-check_xi_c_dot")


if __name__ == '__main__':

    start_time = time.process_time()

    fracture_toughness_primary_sweep = FractureToughnessPrimarySweepCharacterizer()
    fracture_toughness_primary_sweep.characterization()
    # fracture_toughness_primary_sweep.finalization()

    elapsed_wall_time = time.process_time() - start_time
    print(elapsed_wall_time)