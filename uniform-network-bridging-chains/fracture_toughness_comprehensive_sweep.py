"""The single-chain fracture toughness sweep characterization module for
cuFJCs that undergo scission
"""

# import external modules
from __future__ import division
import numpy as np
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


class FractureToughnessSweepCharacterizer(Characterizer):
    """The characterization class assessing fracture toughness for
    composite uFJCs that undergo scission. It inherits all attributes
    and methods from the ``Characterizer`` class.
    """
    def __init__(self):
        """Initializes the ``FractureToughnessSweepCharacterizer`` class
        by initializing and inheriting all attributes and methods from
        the ``Characterizer`` class.
        """
        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        Characterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        p.characterizer.nu_val = 25

        p.characterizer.lmbda_c_eq_min = 0.001
        p.characterizer.lmbda_c_eq_num_steps = 2501

        p.characterizer.xi_c_num_steps = 100001
        p.characterizer.lmbda_c_eq_num_steps = 100001

        p.characterizer.zeta_nu_char_base_val = 100
        p.characterizer.kappa_nu_base_val = 1000

        chain_mech_resp_list = [100, 500, 1000, 5000, 10000]
        chain_mech_resp_dim_sweep_indx_list = [0, 62, 88, 150, 176]

        chain_mech_resp_zeta_nu_char_list = np.copy(chain_mech_resp_list)
        chain_mech_resp_zeta_nu_char_dim_sweep_indx_list = np.copy(
            chain_mech_resp_dim_sweep_indx_list)
        # Reverse the order of the zeta_nu_char list
        chain_mech_resp_zeta_nu_char_list = (
            chain_mech_resp_zeta_nu_char_list[::-1]
        )
        chain_mech_resp_zeta_nu_char_dim_sweep_indx_list = (
            chain_mech_resp_zeta_nu_char_dim_sweep_indx_list[::-1]
        )
        chain_mech_resp_zeta_nu_char_num = (
            len(chain_mech_resp_zeta_nu_char_list)
        )
        chain_mech_resp_zeta_nu_char_label_list = [
            r'$\zeta_{\nu}^{char}='+str(chain_mech_resp_zeta_nu_char_list[chain_mech_resp_zeta_nu_char_indx])+'$'
            for chain_mech_resp_zeta_nu_char_indx
            in range(chain_mech_resp_zeta_nu_char_num)
        ]
        chain_mech_resp_zeta_nu_char_color_list = [
            'orange', 'blue', 'green', 'red', 'purple'
        ]

        p.characterizer.chain_mech_resp_zeta_nu_char_list = (
            chain_mech_resp_zeta_nu_char_list
        )
        p.characterizer.chain_mech_resp_zeta_nu_char_dim_sweep_indx_list = (
            chain_mech_resp_zeta_nu_char_dim_sweep_indx_list
        )
        p.characterizer.chain_mech_resp_zeta_nu_char_num = (
            chain_mech_resp_zeta_nu_char_num
        )
        p.characterizer.chain_mech_resp_zeta_nu_char_label_list = (
            chain_mech_resp_zeta_nu_char_label_list
        )
        p.characterizer.chain_mech_resp_zeta_nu_char_color_list = (
            chain_mech_resp_zeta_nu_char_color_list
        )

        chain_mech_resp_kappa_nu_list = np.copy(chain_mech_resp_list)
        chain_mech_resp_kappa_nu_dim_sweep_indx_list = np.copy(
            chain_mech_resp_dim_sweep_indx_list)
        chain_mech_resp_kappa_nu_num = len(chain_mech_resp_kappa_nu_list)
        chain_mech_resp_kappa_nu_label_list = [
            r'$\kappa_{\nu}='+str(chain_mech_resp_kappa_nu_list[chain_mech_resp_kappa_nu_indx])+'$'
            for chain_mech_resp_kappa_nu_indx
            in range(chain_mech_resp_kappa_nu_num)
        ]
        chain_mech_resp_kappa_nu_color_list = [
            'orange', 'blue', 'green', 'red', 'purple'
        ]

        p.characterizer.chain_mech_resp_kappa_nu_list = (
            chain_mech_resp_kappa_nu_list
        )
        p.characterizer.chain_mech_resp_kappa_nu_dim_sweep_indx_list = (
            chain_mech_resp_kappa_nu_dim_sweep_indx_list
        )
        p.characterizer.chain_mech_resp_kappa_nu_num = (
            chain_mech_resp_kappa_nu_num
        )
        p.characterizer.chain_mech_resp_kappa_nu_label_list = (
            chain_mech_resp_kappa_nu_label_list
        )
        p.characterizer.chain_mech_resp_kappa_nu_color_list = (
            chain_mech_resp_kappa_nu_color_list
        )

        dim_sweep_num = 177

        segment_extensibility_list = np.logspace(2, 4, dim_sweep_num)

        kappa_nu_list = np.copy(segment_extensibility_list)
        kappa_nu_num = len(kappa_nu_list)

        p.characterizer.kappa_nu_list = kappa_nu_list
        p.characterizer.kappa_nu_num = kappa_nu_num

        zeta_nu_char_list = np.copy(segment_extensibility_list)
        zeta_nu_char_num = len(zeta_nu_char_list)

        p.characterizer.zeta_nu_char_list = zeta_nu_char_list
        p.characterizer.zeta_nu_char_num = zeta_nu_char_num

        nondim_rate_prmtr_list = np.logspace(-21, 3, dim_sweep_num)

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

        check_lmbda_c_eq_dot_list = np.copy(nondim_rate_prmtr_list)
        check_lmbda_c_eq_dot_num = len(check_lmbda_c_eq_dot_list)
        check_lmbda_c_eq_dot_list_mpi_split = np.array_split(
            check_lmbda_c_eq_dot_list, self.comm_size)
        check_lmbda_c_eq_dot_num_list_mpi_split = [
            len(check_lmbda_c_eq_dot_list_mpi_split[proc_indx])
            for proc_indx in range(self.comm_size)
        ]
        p.characterizer.check_lmbda_c_eq_dot_list = check_lmbda_c_eq_dot_list
        p.characterizer.check_lmbda_c_eq_dot_num = check_lmbda_c_eq_dot_num
        p.characterizer.check_lmbda_c_eq_dot_list_mpi_split = (
            check_lmbda_c_eq_dot_list_mpi_split
        )
        p.characterizer.check_lmbda_c_eq_dot_num_list_mpi_split = (
            check_lmbda_c_eq_dot_num_list_mpi_split
        )

    def prefix(self):
        """Set characterization prefix"""
        return "fracture_toughness_sweep"
    
    def characterization(self):
        """Define characterization routine"""

        if self.comm_rank == 0:
            print("Fracture toughness sweep characterization")

        T       = 298 # absolute room temperature, K
        k_B     = constants.value(u'Boltzmann constant') # J/K
        h       = constants.value(u'Planck constant') # J/Hz
        hbar    = h / (2*np.pi) # J*sec
        beta    = 1. / (k_B*T) # 1/J
        omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec

        cp = self.parameters.characterizer
        
        check_xi_c_dot_list_mpi_scatter = self.comm.scatter(
            cp.check_xi_c_dot_list_mpi_split, root=0)
        check_lmbda_c_eq_dot_list_mpi_scatter = self.comm.scatter(
            cp.check_lmbda_c_eq_dot_list_mpi_split, root=0)

        # Rate-independent calculations
        
        if self.comm_rank == 0:
            print("Rate-independent calculations")
        
            chain_mech_resp_zeta_nu_char_single_chain_list = [
                RateIndependentScissioncuFJC(
                    nu=cp.nu_val,
                    zeta_nu_char=cp.chain_mech_resp_zeta_nu_char_list[zeta_nu_char_indx],
                    kappa_nu=cp.kappa_nu_base_val)
                for zeta_nu_char_indx
                in range(cp.chain_mech_resp_zeta_nu_char_num)
            ]

            chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk = []
            chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk = []

            for zeta_nu_char_indx in range(cp.chain_mech_resp_zeta_nu_char_num):
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
                    kappa_nu=cp.chain_mech_resp_kappa_nu_list[kappa_nu_indx])
                for kappa_nu_indx in range(cp.chain_mech_resp_kappa_nu_num)
            ]

            chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk = []
            chain_mech_resp_kappa_nu_xi_c___single_chain_chunk = []

            for kappa_nu_indx in range(cp.chain_mech_resp_kappa_nu_num):
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


        # Rate-dependent calculations

        if self.comm_rank == 0:
            print("Rate-dependent calculations")
            print("Rate-dependent check_xi_c_dot versus zeta_nu_char sweep")

        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter = []

        for check_xi_c_dot_val in check_xi_c_dot_list_mpi_scatter:
            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit = []
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_zeta_nu_char_check_tau_c = []
            rate_dependent_zeta_nu_char_log_10_check_tau_c = []

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
                gamma_c_dot = []
                gamma_c_dot_val = 0.
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
                        gamma_c_dot_val = (
                            rate_dependent_single_chain.gamma_c_dot_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[xi_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    gamma_c_dot.append(gamma_c_dot_val)
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char_val
                )

                t_steps_arr = np.asarray(t_steps)
                gamma_c_dot_arr = np.asarray(gamma_c_dot)

                Z_c_sci = np.trapz(gamma_c_dot_arr, t_steps_arr)

                tau_c_gamma_c_rms_intgrnd_arr = gamma_c_dot_arr * t_steps_arr**2

                tau_c_val = (
                    np.sqrt(
                        np.trapz(tau_c_gamma_c_rms_intgrnd_arr, t_steps_arr)/Z_c_sci)
                )

                check_tau_c_val = tau_c_val * omega_0 * cp.nu_val
                log_10_check_tau_c_val = np.log10(check_tau_c_val)

                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
                rate_dependent_zeta_nu_char_check_tau_c.append(
                    check_tau_c_val)
                rate_dependent_zeta_nu_char_log_10_check_tau_c.append(
                    log_10_check_tau_c_val)
            
            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit)
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit)
            rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_check_tau_c)
            rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_log_10_check_tau_c)
        
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
        rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )

        self.comm.Barrier()

        if self.comm_rank == 0:
            print("Rate-dependent check_xi_c_dot versus kappa_nu sweep")

        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter = []

        for check_xi_c_dot_val in check_xi_c_dot_list_mpi_scatter:
            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit = []
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_kappa_nu_check_tau_c = []
            rate_dependent_kappa_nu_log_10_check_tau_c = []

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
                gamma_c_dot = []
                gamma_c_dot_val = 0.
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
                        gamma_c_dot_val = (
                            rate_dependent_single_chain.gamma_c_dot_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[xi_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    gamma_c_dot.append(gamma_c_dot_val)
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / cp.zeta_nu_char_base_val
                )

                t_steps_arr = np.asarray(t_steps)
                gamma_c_dot_arr = np.asarray(gamma_c_dot)

                Z_c_sci = np.trapz(gamma_c_dot_arr, t_steps_arr)

                tau_c_gamma_c_rms_intgrnd_arr = gamma_c_dot_arr * t_steps_arr**2

                tau_c_val = (
                    np.sqrt(
                        np.trapz(tau_c_gamma_c_rms_intgrnd_arr, t_steps_arr)/Z_c_sci)
                )

                check_tau_c_val = tau_c_val * omega_0 * cp.nu_val
                log_10_check_tau_c_val = np.log10(check_tau_c_val)

                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
                rate_dependent_kappa_nu_check_tau_c.append(
                    check_tau_c_val)
                rate_dependent_kappa_nu_log_10_check_tau_c.append(
                    log_10_check_tau_c_val)
            
            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit)
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit)
            rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_check_tau_c)
            rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_log_10_check_tau_c)
        
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
        rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_scatter,
                root=0)
        )

        self.comm.Barrier()

        if self.comm_rank == 0:
            print("Rate-dependent check_lmbda_c_eq_dot versus zeta_nu_char sweep")

        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []
        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []
        rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []
        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []

        for check_lmbda_c_eq_dot_val in check_lmbda_c_eq_dot_list_mpi_scatter:
            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit = []
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_zeta_nu_char_check_tau_c = []
            rate_dependent_zeta_nu_char_log_10_check_tau_c = []

            for zeta_nu_char_val in cp.zeta_nu_char_list:
                rate_dependent_single_chain = (
                    RateDependentScissioncuFJC(nu=cp.nu_val,
                                               zeta_nu_char=zeta_nu_char_val,
                                               kappa_nu=cp.kappa_nu_base_val,
                                               omega_0=omega_0)
                )
                lmbda_c_eq_dot_val = (
                    check_lmbda_c_eq_dot_val * omega_0 * cp.nu_val
                )
                lmbda_c_eq_steps = np.linspace(
                    0, rate_dependent_single_chain.lmbda_c_eq_crit,
                    cp.lmbda_c_eq_num_steps)
                t_steps = (
                    lmbda_c_eq_steps / lmbda_c_eq_dot_val
                ) # 1/(1/sec) = sec
            
                # initialization
                gamma_c_dot = []
                gamma_c_dot_val = 0.
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_steps):
                    t_val = t_steps[lmbda_c_eq_indx]
                    lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_func(
                            lmbda_c_eq_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(
                            lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if lmbda_c_eq_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[lmbda_c_eq_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        gamma_c_dot_val = (
                            rate_dependent_single_chain.gamma_c_dot_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[lmbda_c_eq_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    gamma_c_dot.append(gamma_c_dot_val)
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char_val
                )

                t_steps_arr = np.asarray(t_steps)
                gamma_c_dot_arr = np.asarray(gamma_c_dot)

                Z_c_sci = np.trapz(gamma_c_dot_arr, t_steps_arr)

                tau_c_gamma_c_rms_intgrnd_arr = gamma_c_dot_arr * t_steps_arr**2

                tau_c_val = (
                    np.sqrt(
                        np.trapz(tau_c_gamma_c_rms_intgrnd_arr, t_steps_arr)/Z_c_sci)
                )

                check_tau_c_val = tau_c_val * omega_0 * cp.nu_val
                log_10_check_tau_c_val = np.log10(check_tau_c_val)

                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
                rate_dependent_zeta_nu_char_check_tau_c.append(
                    check_tau_c_val)
                rate_dependent_zeta_nu_char_log_10_check_tau_c.append(
                    log_10_check_tau_c_val)
            
            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit)
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit)
            rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_check_tau_c)
            rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_zeta_nu_char_log_10_check_tau_c)
        
        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )

        self.comm.Barrier()

        if self.comm_rank == 0:
            print("Rate-dependent check_lmbda_c_eq_dot versus kappa_nu sweep")

        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []
        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []
        rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []
        rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter = []

        for check_lmbda_c_eq_dot_val in check_lmbda_c_eq_dot_list_mpi_scatter:
            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit = []
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_kappa_nu_check_tau_c = []
            rate_dependent_kappa_nu_log_10_check_tau_c = []

            for kappa_nu_val in cp.kappa_nu_list:
                rate_dependent_single_chain = (
                    RateDependentScissioncuFJC(nu=cp.nu_val,
                                               zeta_nu_char=cp.zeta_nu_char_base_val,
                                               kappa_nu=kappa_nu_val,
                                               omega_0=omega_0)
                )
                lmbda_c_eq_dot_val = (
                    check_lmbda_c_eq_dot_val * omega_0 * cp.nu_val
                )
                lmbda_c_eq_steps = np.linspace(
                    0, rate_dependent_single_chain.lmbda_c_eq_crit,
                    cp.lmbda_c_eq_num_steps)
                t_steps = (
                    lmbda_c_eq_steps / lmbda_c_eq_dot_val
                ) # 1/(1/sec) = sec
            
                # initialization
                gamma_c_dot = []
                gamma_c_dot_val = 0.
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_steps):
                    t_val = t_steps[lmbda_c_eq_indx]
                    lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_func(
                            lmbda_c_eq_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(
                            lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if lmbda_c_eq_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[lmbda_c_eq_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        gamma_c_dot_val = (
                            rate_dependent_single_chain.gamma_c_dot_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[lmbda_c_eq_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    gamma_c_dot.append(gamma_c_dot_val)
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / cp.zeta_nu_char_base_val
                )

                t_steps_arr = np.asarray(t_steps)
                gamma_c_dot_arr = np.asarray(gamma_c_dot)

                Z_c_sci = np.trapz(gamma_c_dot_arr, t_steps_arr)

                tau_c_gamma_c_rms_intgrnd_arr = gamma_c_dot_arr * t_steps_arr**2

                tau_c_val = (
                    np.sqrt(
                        np.trapz(tau_c_gamma_c_rms_intgrnd_arr, t_steps_arr)/Z_c_sci)
                )

                check_tau_c_val = tau_c_val * omega_0 * cp.nu_val
                log_10_check_tau_c_val = np.log10(check_tau_c_val)

                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
                rate_dependent_kappa_nu_check_tau_c.append(
                    check_tau_c_val)
                rate_dependent_kappa_nu_log_10_check_tau_c.append(
                    log_10_check_tau_c_val)
            
            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit)
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit)
            rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_check_tau_c)
            rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter.append(
                rate_dependent_kappa_nu_log_10_check_tau_c)
        
        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_scatter,
                root=0)
        )

        self.comm.Barrier()


        if self.comm_rank == 0:
            print("Post-processing rate-dependent calculations")

            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []
            rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list = []
            rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list = []

            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = []
            rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list = []
            rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list = []

            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = []
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = []
            rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list = []
            rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list = []

            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = []
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = []
            rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list = []
            rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list = []

            for proc_indx in range(self.comm_size):
                for check_xi_c_dot_chunk_indx in range(cp.check_xi_c_dot_num_list_mpi_split[proc_indx]):
                    rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )

                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val = (
                        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_val = (
                        rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )
                    rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_val = (
                        rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    )

                    rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)
                    rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_val)
                    rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_val)

                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list.append(
                        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_val)
                    rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list.append(
                        rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_val)
                    rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list.append(
                        rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_val)
                
                for check_lmbda_c_eq_dot_chunk_indx in range(cp.check_lmbda_c_eq_dot_num_list_mpi_split[proc_indx]):
                    rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )
                    rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )
                    rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )

                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )
                    rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )
                    rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_val = (
                        rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list_mpi_split[proc_indx][check_lmbda_c_eq_dot_chunk_indx]
                    )

                    rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val)
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val)
                    rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_val)
                    rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_val)

                    rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val)
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_val)
                    rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_val)
                    rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list.append(
                        rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_val)
            
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
                rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list,
                "rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list,
                "rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list")
            
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
                rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list,
                "rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list,
                "rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list,
                "rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list")

    def finalization(self):
        """Define finalization analysis"""

        if self.comm_rank == 0:
            print("Fracture toughness sweep finalization")

        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing
        
        if self.comm_rank == 0:
            print("Plotting")

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

            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            )
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            )
            rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_check_tau_c___check_xi_c_dot_chunk_list")
            )
            rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_log_10_check_tau_c___check_xi_c_dot_chunk_list")
            )

            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            )
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list")
            )
            rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_check_tau_c___check_xi_c_dot_chunk_list")
            )
            rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_log_10_check_tau_c___check_xi_c_dot_chunk_list")
            )

            rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            )
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            )
            rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_check_tau_c___check_lmbda_c_eq_dot_chunk_list")
            )
            rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_zeta_nu_char_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list")
            )

            rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            )
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list")
            )
            rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_check_tau_c___check_lmbda_c_eq_dot_chunk_list")
            )
            rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    "rate_dependent_kappa_nu_log_10_check_tau_c___check_lmbda_c_eq_dot_chunk_list")
            )

            # plot results
            latex_formatting_figure(ppp)

            fig = plt.figure()
            for chain_mech_resp_kappa_nu_indx \
                in range(cp.chain_mech_resp_kappa_nu_num):
                lmbda_c_eq_list = (
                    chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk[chain_mech_resp_kappa_nu_indx]
                )
                xi_c_list = (
                    chain_mech_resp_kappa_nu_xi_c___single_chain_chunk[chain_mech_resp_kappa_nu_indx]
                )
                plt.semilogy(lmbda_c_eq_list, xi_c_list, linestyle='-',
                    color=cp.chain_mech_resp_kappa_nu_color_list[chain_mech_resp_kappa_nu_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.chain_mech_resp_kappa_nu_label_list[chain_mech_resp_kappa_nu_indx])
                plt.plot(lmbda_c_eq_list[-1], xi_c_list[-1], marker='x',
                    color=cp.chain_mech_resp_kappa_nu_color_list[chain_mech_resp_kappa_nu_indx],
                    alpha=1, linewidth=2.5)
            plt.legend(loc='best', fontsize=10)
            plt.xlim([-0.05, 2.05])
            plt.xticks(fontsize=20)
            plt.ylim([1e-2, 1e4])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_c$', 30,
                "kappa_nu-xi_c-vs-lmbda_c_eq")
            
            fig = plt.figure()
            for chain_mech_resp_zeta_nu_char_indx \
                in range(cp.chain_mech_resp_zeta_nu_char_num):
                lmbda_c_eq_list = chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk[chain_mech_resp_zeta_nu_char_indx]
                xi_c_list = chain_mech_resp_zeta_nu_char_xi_c___single_chain_chunk[chain_mech_resp_zeta_nu_char_indx]
                plt.semilogy(lmbda_c_eq_list, xi_c_list, linestyle='-',
                    color=cp.chain_mech_resp_zeta_nu_char_color_list[chain_mech_resp_zeta_nu_char_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.chain_mech_resp_zeta_nu_char_label_list[chain_mech_resp_zeta_nu_char_indx])
                plt.plot(lmbda_c_eq_list[-1], xi_c_list[-1], marker='x',
                    color=cp.chain_mech_resp_zeta_nu_char_color_list[chain_mech_resp_zeta_nu_char_indx],
                    alpha=1, linewidth=2.5)
            plt.legend(loc='best', fontsize=10)
            plt.xlim([-0.05, 4.25])
            plt.xticks(fontsize=20)
            plt.ylim([1e-2, 1e4])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_c$', 30,
                "zeta_nu_char-xi_c-vs-lmbda_c_eq")
            
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list)
            )
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr = (
                np.transpose(
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )
            
            fig = plt.figure()
            for chain_mech_resp_kappa_nu_indx \
                in range(cp.chain_mech_resp_kappa_nu_num):
                chain_mech_resp_kappa_nu_dim_sweep_indx = (
                    cp.chain_mech_resp_kappa_nu_dim_sweep_indx_list[chain_mech_resp_kappa_nu_indx]
                )
                rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit_list = (
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr[chain_mech_resp_kappa_nu_dim_sweep_indx]
                )
                plt.semilogx(
                    cp.check_xi_c_dot_list,
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-',
                    color=cp.chain_mech_resp_kappa_nu_color_list[chain_mech_resp_kappa_nu_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.chain_mech_resp_kappa_nu_label_list[chain_mech_resp_kappa_nu_indx])
            plt.legend(loc='best', fontsize=10)
            plt.xlim([1e-21, 1e0])
            plt.xticks([1e-21, 1e-14, 1e-7, 1e0], fontsize=20)
            plt.ylim([-0.01, 0.51])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\check{\dot{\xi}}_c$', 30,
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$', 30,
                "kappa_nu-overline_epsilon_cnu_diss_hat_crit-vs-check_xi_c_dot")

            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list)
            )
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr = (
                np.transpose(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )
            
            fig = plt.figure()
            for chain_mech_resp_zeta_nu_char_indx \
                in range(cp.chain_mech_resp_zeta_nu_char_num):
                chain_mech_resp_zeta_nu_char_dim_sweep_indx = (
                    cp.chain_mech_resp_zeta_nu_char_dim_sweep_indx_list[chain_mech_resp_zeta_nu_char_indx]
                )
                rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit_list = (
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr[chain_mech_resp_zeta_nu_char_dim_sweep_indx]
                )
                plt.semilogx(
                    cp.check_xi_c_dot_list,
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-',
                    color=cp.chain_mech_resp_zeta_nu_char_color_list[chain_mech_resp_zeta_nu_char_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.chain_mech_resp_zeta_nu_char_label_list[chain_mech_resp_zeta_nu_char_indx])
            plt.legend(loc='best', fontsize=10)
            plt.xlim([1e-21, 1e0])
            plt.xticks([1e-21, 1e-14, 1e-7, 1e0], fontsize=20)
            plt.ylim([-0.01, 0.51])
            plt.yticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\check{\dot{\xi}}_c$', 30,
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$', 30,
                "zeta_nu_char-overline_epsilon_cnu_diss_hat_crit-vs-check_xi_c_dot")

            overline_epsilon_cnu_diss_hat_crit_contourf_levels_num = 101
            overline_epsilon_cnu_diss_hat_crit_contourf_levels = np.linspace(
                0, 1, overline_epsilon_cnu_diss_hat_crit_contourf_levels_num)
            
            overline_epsilon_cnu_diss_hat_crit_contourf_ticks_num = 11
            overline_epsilon_cnu_diss_hat_crit_contourf_ticks = np.linspace(
                0, 1, overline_epsilon_cnu_diss_hat_crit_contourf_ticks_num)

            overline_epsilon_cnu_diss_hat_crit_contour_levels_num = 26
            overline_epsilon_cnu_diss_hat_crit_contour_levels = np.linspace(
                0, 1, overline_epsilon_cnu_diss_hat_crit_contour_levels_num)
            
            colormap = plt.cm.hsv


            kappa_nu_list_meshgrid, check_xi_c_dot_list_meshgrid = np.meshgrid(
                cp.kappa_nu_list, cp.check_xi_c_dot_list)
            
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list)
            )
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr = (
                np.transpose(
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )
            
            init_mask_indx_list = []

            for kappa_nu_indx in range(cp.kappa_nu_num):
                max_check_xi_c_dot_indx = np.argmax(
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr[kappa_nu_indx])
                mask_indx = max_check_xi_c_dot_indx + 1
                init_mask_indx_list.append(mask_indx)
            
            premasked_rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr = (
                np.copy(
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr)
            )

            for kappa_nu_indx in range(cp.kappa_nu_num):
                init_mask_indx = init_mask_indx_list[kappa_nu_indx]
                for mask_indx in range(init_mask_indx, cp.check_xi_c_dot_num):
                    premasked_rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr[kappa_nu_indx][mask_indx] = (
                        np.nan
                    )
            
            premasked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.transpose(
                    premasked_rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr)
            )

            masked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.ma.masked_invalid(
                    premasked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )

            fig, ax1 = plt.subplots()

            ax1.patch.set_facecolor('silver')

            filled_contour_plot = ax1.contourf(
                kappa_nu_list_meshgrid, check_xi_c_dot_list_meshgrid,
                masked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contourf_levels,
                cmap=colormap)
            
            for fcp in filled_contour_plot.collections:
                fcp.set_edgecolor('face')
            
            ax1.set_xlabel(r'$\kappa_{\nu}$', fontsize=30)
            ax1.set_ylabel(r'$\check{\dot{\xi}}_c$', fontsize=30)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_yticks(np.logspace(-21, 3, 5))
            # ax1.tick_params(axis='both', labelsize=16)

            labeled_contour_plot = ax1.contour(
                kappa_nu_list_meshgrid, check_xi_c_dot_list_meshgrid,
                masked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contour_levels,
                colors=('black',), linewidths=0.25)
            ax1.clabel(
                labeled_contour_plot, fmt='%3.2f',
                colors='black',  fontsize=10)
            
            cbar = fig.colorbar(
                filled_contour_plot,
                ticks=overline_epsilon_cnu_diss_hat_crit_contourf_ticks)
            cbar.ax.set_ylabel(
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$',
                fontsize=30)
            cbar.ax.tick_params(axis='y', labelsize=20)
            
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)

            plt.tight_layout()
            fig.savefig(
                self.savedir+"overline_epsilon_cnu_diss_hat_crit-filled-contour-check_xi_c_dot-vs-kappa_nu"+".pdf",
                transparent=False)
            plt.close()

            kappa_nu_list_meshgrid, check_lmbda_c_eq_dot_list_meshgrid = np.meshgrid(
                cp.kappa_nu_list, cp.check_lmbda_c_eq_dot_list)
            
            rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list)
            )
            rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr = (
                np.transpose(
                    rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr)
            )
            
            init_mask_indx_list = []

            for kappa_nu_indx in range(cp.kappa_nu_num):
                max_check_lmbda_c_eq_dot_indx = np.argmax(
                    rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr[kappa_nu_indx])
                mask_indx = max_check_lmbda_c_eq_dot_indx + 1
                init_mask_indx_list.append(mask_indx)
            
            premasked_rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr = (
                np.copy(
                    rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr)
            )

            for kappa_nu_indx in range(cp.kappa_nu_num):
                init_mask_indx = init_mask_indx_list[kappa_nu_indx]
                for mask_indx \
                    in range(init_mask_indx, cp.check_lmbda_c_eq_dot_num):
                    premasked_rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr[kappa_nu_indx][mask_indx] = (
                        np.nan
                    )
            
            premasked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr = (
                np.transpose(
                    premasked_rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___kappa_nu_chunk_arr)
            )

            masked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr = (
                np.ma.masked_invalid(
                    premasked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr)
            )

            fig, ax1 = plt.subplots()

            ax1.patch.set_facecolor('silver')

            filled_contour_plot = ax1.contourf(
                kappa_nu_list_meshgrid, check_lmbda_c_eq_dot_list_meshgrid,
                masked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contourf_levels,
                cmap=colormap)
            
            for fcp in filled_contour_plot.collections:
                fcp.set_edgecolor('face')
            
            ax1.set_xlabel(r'$\kappa_{\nu}$', fontsize=30)
            ax1.set_ylabel(r'$\check{\dot{\lambda}}_c^{eq}$', fontsize=30)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(top=1)
            ax1.set_yticks(np.logspace(-21, 0, 4))
            # ax1.tick_params(axis='both', labelsize=16)

            labeled_contour_plot = ax1.contour(
                kappa_nu_list_meshgrid, check_lmbda_c_eq_dot_list_meshgrid,
                masked_rate_dependent_kappa_nu_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contour_levels,
                colors=('black',), linewidths=0.25)
            ax1.clabel(
                labeled_contour_plot, fmt='%3.2f',
                colors='black',  fontsize=10)
            
            cbar = fig.colorbar(
                filled_contour_plot,
                ticks=overline_epsilon_cnu_diss_hat_crit_contourf_ticks)
            cbar.ax.set_ylabel(
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$',
                fontsize=30)
            cbar.ax.tick_params(axis='y', labelsize=20)
            
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)

            plt.tight_layout()
            fig.savefig(
                self.savedir+"overline_epsilon_cnu_diss_hat_crit-filled-contour-check_lmbda_c_eq_dot-vs-kappa_nu"+".pdf",
                transparent=False)
            plt.close()

            zeta_nu_char_list_meshgrid, check_xi_c_dot_list_meshgrid = np.meshgrid(
                cp.zeta_nu_char_list, cp.check_xi_c_dot_list)
            
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_list)
            )
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr = (
                np.transpose(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )
            
            init_mask_indx_list = []

            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                max_check_xi_c_dot_indx = np.argmax(
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr[zeta_nu_char_indx])
                mask_indx = max_check_xi_c_dot_indx + 1
                init_mask_indx_list.append(mask_indx)
            
            premasked_rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr = (
                np.copy(
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr)
            )

            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                init_mask_indx = init_mask_indx_list[zeta_nu_char_indx]
                for mask_indx in range(init_mask_indx, cp.check_xi_c_dot_num):
                    premasked_rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr[zeta_nu_char_indx][mask_indx] = (
                        np.nan
                    )
            
            premasked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.transpose(
                    premasked_rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr)
            )

            masked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr = (
                np.ma.masked_invalid(
                    premasked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr)
            )

            fig, ax1 = plt.subplots()

            ax1.patch.set_facecolor('silver')

            filled_contour_plot = ax1.contourf(
                zeta_nu_char_list_meshgrid, check_xi_c_dot_list_meshgrid,
                masked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contourf_levels,
                cmap=colormap)
            
            for fcp in filled_contour_plot.collections:
                fcp.set_edgecolor('face')
            
            ax1.set_xlabel(r'$\zeta_{\nu}^{char}$', fontsize=30)
            ax1.set_ylabel(r'$\check{\dot{\xi}}_c$', fontsize=30)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_yticks(np.logspace(-21, 3, 5))
            # ax1.tick_params(axis='both', labelsize=16)

            labeled_contour_plot = ax1.contour(
                zeta_nu_char_list_meshgrid, check_xi_c_dot_list_meshgrid,
                masked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_xi_c_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contour_levels,
                colors=('black',), linewidths=0.25)
            ax1.clabel(
                labeled_contour_plot, fmt='%3.2f',
                colors='black',  fontsize=10)
            
            cbar = fig.colorbar(
                filled_contour_plot,
                ticks=overline_epsilon_cnu_diss_hat_crit_contourf_ticks)
            cbar.ax.set_ylabel(
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$',
                fontsize=30)
            cbar.ax.tick_params(axis='y', labelsize=20)
            
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)

            plt.tight_layout()
            fig.savefig(
                self.savedir+"overline_epsilon_cnu_diss_hat_crit-filled-contour-check_xi_c_dot-vs-zeta_nu_char"+".pdf",
                transparent=False)
            plt.close()

            zeta_nu_char_list_meshgrid, check_lmbda_c_eq_dot_list_meshgrid = np.meshgrid(
                cp.zeta_nu_char_list, cp.check_lmbda_c_eq_dot_list)
            
            rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr = (
                np.asarray(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_list)
            )
            rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr = (
                np.transpose(
                    rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr)
            )
            
            init_mask_indx_list = []

            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                max_check_lmbda_c_eq_dot_indx = np.argmax(
                    rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr[zeta_nu_char_indx])
                mask_indx = max_check_lmbda_c_eq_dot_indx + 1
                init_mask_indx_list.append(mask_indx)
            
            premasked_rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr = (
                np.copy(
                    rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr)
            )

            for zeta_nu_char_indx in range(cp.zeta_nu_char_num):
                init_mask_indx = init_mask_indx_list[zeta_nu_char_indx]
                for mask_indx \
                    in range(init_mask_indx, cp.check_lmbda_c_eq_dot_num):
                    premasked_rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr[zeta_nu_char_indx][mask_indx] = (
                        np.nan
                    )
            
            premasked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr = (
                np.transpose(
                    premasked_rate_dependent_check_lmbda_c_eq_dot_overline_epsilon_cnu_diss_hat_crit___zeta_nu_char_chunk_arr)
            )

            masked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr = (
                np.ma.masked_invalid(
                    premasked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr)
            )

            fig, ax1 = plt.subplots()

            ax1.patch.set_facecolor('silver')

            filled_contour_plot = ax1.contourf(
                zeta_nu_char_list_meshgrid, check_lmbda_c_eq_dot_list_meshgrid,
                masked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contourf_levels,
                cmap=colormap)
            
            for fcp in filled_contour_plot.collections:
                fcp.set_edgecolor('face')
            
            ax1.set_xlabel(r'$\zeta_{\nu}^{char}$', fontsize=30)
            ax1.set_ylabel(r'$\check{\dot{\lambda}}_c^{eq}$', fontsize=30)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(top=1)
            ax1.set_yticks(np.logspace(-21, 0, 4))
            # ax1.tick_params(axis='both', labelsize=16)

            labeled_contour_plot = ax1.contour(
                zeta_nu_char_list_meshgrid, check_lmbda_c_eq_dot_list_meshgrid,
                masked_rate_dependent_zeta_nu_char_overline_epsilon_cnu_diss_hat_crit___check_lmbda_c_eq_dot_chunk_arr,
                levels=overline_epsilon_cnu_diss_hat_crit_contour_levels,
                colors=('black',), linewidths=0.25)
            ax1.clabel(
                labeled_contour_plot, fmt='%3.2f',
                colors='black',  fontsize=10)
            
            cbar = fig.colorbar(
                filled_contour_plot,
                ticks=overline_epsilon_cnu_diss_hat_crit_contourf_ticks)
            cbar.ax.set_ylabel(
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$',
                fontsize=30)
            cbar.ax.tick_params(axis='y', labelsize=20)
            
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)

            plt.tight_layout()
            fig.savefig(
                self.savedir+"overline_epsilon_cnu_diss_hat_crit-filled-contour-check_lmbda_c_eq_dot-vs-zeta_nu_char"+".pdf",
                transparent=False)
            plt.close()


if __name__ == '__main__':

    start_time = time.process_time()

    fracture_toughness_sweep = FractureToughnessSweepCharacterizer()
    fracture_toughness_sweep.characterization()
    # fracture_toughness_sweep.finalization()

    elapsed_wall_time = time.process_time() - start_time
    print(elapsed_wall_time)