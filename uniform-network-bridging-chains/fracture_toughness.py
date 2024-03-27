"""The single-chain fracture toughness characterization module for
cuFJCs that undergo scission
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
    save_current_figure_no_labels,
    save_pickle_object,
    load_pickle_object
)


class FractureToughnessCharacterizer(Characterizer):
    """The characterization class assessing fracture toughness for
    composite uFJCs that undergo scission. It inherits all attributes
    and methods from the ``Characterizer`` class.
    """
    def __init__(self, paper_authors, chain, T):
        """Initializes the ``FractureToughnessCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``Characterizer`` class.
        """
        self.paper_authors = paper_authors
        self.chain = chain
        self.T = T

        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        Characterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        p.characterizer.chain_data_directory = (
            "./AFM_chain_tensile_test_curve_fit_results/"
        )

        p.characterizer.paper_authors2polymer_type_dict = {
            "al-maawali-et-al": "pdms",
            "hugel-et-al": "pva"
        }
        p.characterizer.paper_authors2polymer_type_label_dict = {
            "al-maawali-et-al": r'$\textrm{PDMS single chain data}$',
            "hugel-et-al": r'$\textrm{PVA single chain data}$'
        }
        p.characterizer.polymer_type_label2chain_backbone_bond_type_dict = {
            "pdms": "si-o",
            "pva": "c-c"
        }

        p.characterizer.cufjc_A_nu_linestyle = '-'
        p.characterizer.inext_gaussian_A_nu_linestyle = '--'

        p.characterizer.LT_full_zeta_nu_char_label = (
            r'$\textrm{LT}~(\hat{\varepsilon}_{c\nu}^{sci})^{crit} = \zeta_{\nu}^{char}$'
        )
        p.characterizer.LT_overline_full_zeta_nu_char_label = (
            r'$\textrm{LT}~\overline{(\hat{\varepsilon}_{c\nu}^{sci})^{crit}} = 1$'
        )
        p.characterizer.LT_full_zeta_nu_char_color = 'red'
        p.characterizer.LT_full_zeta_nu_char_linestyle = '--'

        p.characterizer.LT_quarter_zeta_nu_char_label = (
            r'$\textrm{LT}~(\hat{\varepsilon}_{c\nu}^{sci})^{crit} = \zeta_{\nu}^{char}/4$'
        )
        p.characterizer.LT_overline_quarter_zeta_nu_char_label = (
            r'$\textrm{LT}~\overline{(\hat{\varepsilon}_{c\nu}^{sci})^{crit}} = 0.25$'
        )
        p.characterizer.LT_quarter_zeta_nu_char_color = 'magenta'
        p.characterizer.LT_quarter_zeta_nu_char_linestyle = '--'

        p.characterizer.cufjc_label = (
            r'$\textrm{c}u\textrm{FJC scission}~(\hat{\varepsilon}_{c\nu}^{diss})^{crit}$'
        )
        p.characterizer.cufjc_overline_label = (
            r'$\textrm{c}u\textrm{FJC scission}~\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$'
        )
        p.characterizer.cufjc_color = 'blue'
        p.characterizer.cufjc_linestyle = '-'

        p.characterizer.f_c_num_steps = 100001
        p.characterizer.r_nu_num_steps = 100001

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

        # from DFT simulations on H_3C-CH_2-CH_3 (c-c) 
        # and H_3Si-O-CH_3 (si-o) by Beyer, J Chem. Phys., 2000
        chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict = {
            "c-c": [6.05, 4.72, 3.81, 3.07, 2.45],
            "si-o": [4.77, 3.79, 3.14, 2.61, 2.17]
        } # nN
        beyer_2000_tau_b_list = [1e-12, 1e-6, 1e0, 1e5, 1e12] # sec
        beyer_2000_tau_b_num = len(beyer_2000_tau_b_list)
        beyer_2000_tau_b_exponent_list = [
            int(floor(log10(abs(beyer_2000_tau_b_list[tau_b_indx]))))
            for tau_b_indx in range(beyer_2000_tau_b_num)
        ]
        beyer_2000_tau_b_label_list = [
            r'$\textrm{WPRC}~(\hat{\varepsilon}_{c\nu}^{sci})^{crit},~\tau_{\nu}='+'10^{%g}~sec$' % (beyer_2000_tau_b_exponent_list[tau_b_indx])
            for tau_b_indx in range(beyer_2000_tau_b_num)
        ]
        beyer_2000_tau_b_overline_label_list = [
            r'$\textrm{WPRC}~\overline{(\hat{\varepsilon}_{c\nu}^{sci})^{crit}},~\tau_{\nu}='+'10^{%g}~sec$' % (beyer_2000_tau_b_exponent_list[tau_b_indx])
            for tau_b_indx in range(beyer_2000_tau_b_num)
        ]
        beyer_2000_tau_b_color_list = ['gainsboro', 'lightgrey', 'silver', 'darkgrey', 'grey']
        beyer_2000_tau_b_linestyle_list = ['--', '--', '--', '--', '--']

        p.characterizer.chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict = (
            chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict
        )
        p.characterizer.beyer_2000_tau_b_list = beyer_2000_tau_b_list
        p.characterizer.beyer_2000_tau_b_num = beyer_2000_tau_b_num
        p.characterizer.beyer_2000_tau_b_exponent_list = (
            beyer_2000_tau_b_exponent_list
        )
        p.characterizer.beyer_2000_tau_b_label_list = (
            beyer_2000_tau_b_label_list
        )
        p.characterizer.beyer_2000_tau_b_overline_label_list = (
            beyer_2000_tau_b_overline_label_list
        )
        p.characterizer.beyer_2000_tau_b_color_list = (
            beyer_2000_tau_b_color_list
        )
        p.characterizer.beyer_2000_tau_b_linestyle_list = (
            beyer_2000_tau_b_linestyle_list
        )

        p.characterizer.AFM_exprmts_indx_list = [1, 2, 3]
        p.characterizer.typcl_AFM_exprmt_indx = 2

        f_c_dot_list = [1e1, 1e5, 1e9] # nN/sec
        f_c_dot_list = f_c_dot_list[::-1] # reverse order
        f_c_dot_num = len(f_c_dot_list)
        f_c_dot_exponent_list = [
            int(floor(log10(abs(f_c_dot_list[f_c_dot_indx]))))
            for f_c_dot_indx in range(f_c_dot_num)
        ]
        f_c_dot_label_list = [
            r'$\textrm{c}u\textrm{FJC scission}~(\hat{\varepsilon}_{c\nu}^{diss})^{crit},~\dot{f}_c='+'10^{%g}~nN/sec$' % (f_c_dot_exponent_list[f_c_dot_indx])
            for f_c_dot_indx in range(f_c_dot_num)
        ]
        f_c_dot_overline_label_list = [
            r'$\textrm{c}u\textrm{FJC scission}~\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}},~\dot{f}_c='+'10^{%g}~nN/sec$' % (f_c_dot_exponent_list[f_c_dot_indx])
            for f_c_dot_indx in range(f_c_dot_num)
        ]
        f_c_dot_comparison_label_list = [
            r'$\textrm{c}u\textrm{FJC scission},~\dot{f}_c='+'10^{%g}~nN/sec$' % (f_c_dot_exponent_list[f_c_dot_indx])
            for f_c_dot_indx in range(f_c_dot_num)
        ]
        f_c_dot_comparison_overline_label_list = [
            r'$\textrm{c}u\textrm{FJC scission},~\dot{f}_c='+'10^{%g}~nN/sec$' % (f_c_dot_exponent_list[f_c_dot_indx])
            for f_c_dot_indx in range(f_c_dot_num)
        ]
        f_c_dot_color_list = ['orange', 'purple', 'green']
        f_c_dot_color_list = f_c_dot_color_list[::-1]
        f_c_dot_linestyle_list = ['-', '-', '-']
        f_c_dot_linestyle_list = f_c_dot_linestyle_list[::-1]
        f_c_dot_comparison_color_list = ['orange', 'purple', 'green']
        f_c_dot_comparison_color_list = f_c_dot_comparison_color_list[::-1]
        f_c_dot_comparison_linestyle_list = ['-', '-', '-']
        f_c_dot_comparison_linestyle_list = (
            f_c_dot_comparison_linestyle_list[::-1]
        )

        p.characterizer.f_c_dot_list = f_c_dot_list
        p.characterizer.f_c_dot_num = f_c_dot_num
        p.characterizer.f_c_dot_exponent_list = f_c_dot_exponent_list
        p.characterizer.f_c_dot_label_list = f_c_dot_label_list
        p.characterizer.f_c_dot_overline_label_list = (
            f_c_dot_overline_label_list
        )
        p.characterizer.f_c_dot_comparison_label_list = (
            f_c_dot_comparison_label_list
        )
        p.characterizer.f_c_dot_comparison_overline_label_list = (
            f_c_dot_comparison_overline_label_list
        )
        p.characterizer.f_c_dot_color_list = f_c_dot_color_list
        p.characterizer.f_c_dot_linestyle_list = f_c_dot_linestyle_list
        p.characterizer.f_c_dot_comparison_color_list = (
            f_c_dot_comparison_color_list
        )
        p.characterizer.f_c_dot_comparison_linestyle_list = (
            f_c_dot_comparison_linestyle_list
        )

        r_nu_dot_list = [1e1, 1e5, 1e9] # nm/sec
        r_nu_dot_list = r_nu_dot_list[::-1] # reverse order
        r_nu_dot_num = len(r_nu_dot_list)
        r_nu_dot_exponent_list = [
            int(floor(log10(abs(r_nu_dot_list[r_nu_dot_indx]))))
            for r_nu_dot_indx in range(r_nu_dot_num)
        ]
        r_nu_dot_label_list = [
            r'$\textrm{c}u\textrm{FJC scission}~(\hat{\varepsilon}_{c\nu}^{diss})^{crit},~\dot{r}_{\nu}='+'10^{%g}~nm/sec$' % (r_nu_dot_exponent_list[r_nu_dot_indx])
            for r_nu_dot_indx in range(r_nu_dot_num)
        ]
        r_nu_dot_overline_label_list = [
            r'$\textrm{c}u\textrm{FJC scission}~\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}},~\dot{r}_{\nu}='+'10^{%g}~nm/sec$' % (r_nu_dot_exponent_list[r_nu_dot_indx])
            for r_nu_dot_indx in range(r_nu_dot_num)
        ]
        r_nu_dot_comparison_label_list = [
            r'$\textrm{c}u\textrm{FJC scission},~\dot{r}_{\nu}='+'10^{%g}~nm/sec$' % (r_nu_dot_exponent_list[r_nu_dot_indx])
            for r_nu_dot_indx in range(r_nu_dot_num)
        ]
        r_nu_dot_comparison_overline_label_list = [
            r'$\textrm{c}u\textrm{FJC scission},~\dot{r}_{\nu}='+'10^{%g}~nm/sec$' % (r_nu_dot_exponent_list[r_nu_dot_indx])
            for r_nu_dot_indx in range(r_nu_dot_num)
        ]
        r_nu_dot_color_list = ['orange', 'purple', 'green']
        r_nu_dot_color_list = r_nu_dot_color_list[::-1]
        r_nu_dot_linestyle_list = ['-', '-', '-']
        r_nu_dot_linestyle_list = r_nu_dot_linestyle_list[::-1]
        r_nu_dot_comparison_color_list = ['orange', 'purple', 'green']
        r_nu_dot_comparison_color_list = r_nu_dot_comparison_color_list[::-1]
        r_nu_dot_comparison_linestyle_list = ['-.', '-.', '-.']
        r_nu_dot_comparison_linestyle_list = (
            r_nu_dot_comparison_linestyle_list[::-1]
        )

        p.characterizer.r_nu_dot_list = r_nu_dot_list
        p.characterizer.r_nu_dot_num = r_nu_dot_num
        p.characterizer.r_nu_dot_exponent_list = r_nu_dot_exponent_list
        p.characterizer.r_nu_dot_label_list = r_nu_dot_label_list
        p.characterizer.r_nu_dot_overline_label_list = (
            r_nu_dot_overline_label_list
        )
        p.characterizer.r_nu_dot_comparison_label_list = (
            r_nu_dot_comparison_label_list
        )
        p.characterizer.r_nu_dot_comparison_overline_label_list = (
            r_nu_dot_comparison_overline_label_list
        )
        p.characterizer.r_nu_dot_color_list = r_nu_dot_color_list
        p.characterizer.r_nu_dot_linestyle_list = r_nu_dot_linestyle_list
        p.characterizer.r_nu_dot_comparison_color_list = (
            r_nu_dot_comparison_color_list
        )
        p.characterizer.r_nu_dot_comparison_linestyle_list = (
            r_nu_dot_comparison_linestyle_list
        )

    def prefix(self):
        """Set characterization prefix"""
        return "fracture_toughness"
    
    def characterization(self):
        """Define characterization routine"""

        if self.comm_rank == 0:
            print(self.paper_authors+" "+self.chain+" characterization")

        k_B = constants.value(u'Boltzmann constant') # J/K
        h = constants.value(u'Planck constant') # J/Hz
        hbar = h / (2*np.pi) # J*sec
        beta = 1. / (k_B*self.T) # 1/J
        omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec

        beta = beta / (1e9*1e9) # 1/J = 1/(N*m) -> 1/(nN*m) -> 1/(nN*nm)

        cp = self.parameters.characterizer

        polymer_type = cp.paper_authors2polymer_type_dict[self.paper_authors]
        chain_backbone_bond_type = (
            cp.polymer_type_label2chain_backbone_bond_type_dict[polymer_type]
        )
        data_file_prefix = (
            self.paper_authors + '-' + polymer_type + '-'
            + chain_backbone_bond_type + '-' + self.chain
        )

        # unitless, unitless, unitless, nm, respectively
        nu = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-intgr_nu'+'.txt')
        zeta_nu_char = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt')
        kappa_nu = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-kappa_nu_intgr_nu'+'.txt')
        l_nu_eq = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-l_nu_eq_intgr_nu'+'.txt')
        
        nu_list_mpi_scatter = self.comm.scatter(cp.nu_list_mpi_split, root=0)
        
        rate_independent_AFM_exprmt_single_chain = (
            RateIndependentScissioncuFJC(nu=nu,
                                         zeta_nu_char=zeta_nu_char,
                                         kappa_nu=kappa_nu)
        )
        
        beyer_2000_f_c_max_tau_b_list = (
            cp.chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict[chain_backbone_bond_type]
        ) # nN


        # Rate-independent calculations
        if self.comm_rank == 0:
            print("Rate-independent calculations")
        
        A_nu___nu_chunk_list_mpi_scatter = []
        inext_gaussian_A_nu___nu_chunk_list_mpi_scatter = []
        inext_gaussian_A_nu_err___nu_chunk_list_mpi_scatter = []
        
        rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_g_c___nu_chunk_list_mpi_scatter = []
        rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_overline_g_c___nu_chunk_list_mpi_scatter = []
        
        rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_scatter = []
        
        rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter = []

        rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_scatter = []
        
        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_scatter = []
        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter = []
        
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list_mpi_scatter = []

        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_g_c = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c = []

            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c = []
            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c = []

            rate_independent_single_chain = (
                RateIndependentScissioncuFJC(nu=nu_val,
                                             zeta_nu_char=zeta_nu_char,
                                             kappa_nu=kappa_nu)
            )

            A_nu_val = rate_independent_single_chain.A_nu
            inext_gaussian_A_nu_val = 1 / np.sqrt(nu_val)
            inext_gaussian_A_nu_err_val = (
                np.abs((inext_gaussian_A_nu_val-A_nu_val)/A_nu_val) * 100
            )

            rate_independent_epsilon_cnu_diss_hat_crit_val = (
                rate_independent_single_chain.epsilon_cnu_diss_hat_crit
            )
            rate_independent_epsilon_c_diss_hat_crit_val = (
                nu_val * rate_independent_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_g_c_val = rate_independent_single_chain.g_c_crit
            rate_independent_overline_epsilon_cnu_diss_hat_crit_val = (
                rate_independent_epsilon_cnu_diss_hat_crit_val / zeta_nu_char
            )
            rate_independent_overline_epsilon_c_diss_hat_crit_val = (
                nu_val * rate_independent_overline_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_overline_g_c_val = (
                rate_independent_g_c_val / zeta_nu_char
            )

            rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit_val = (
                zeta_nu_char
            )
            rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit_val = (
                nu_val
                * rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_full_zeta_nu_char_g_c_val = (
                0.5 * A_nu_val * nu_val
                * rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val = (
                1.
            )
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit_val = (
                nu_val
                * rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_full_zeta_nu_char_overline_g_c_val = (
                0.5 * A_nu_val * nu_val
                * rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val
            )

            rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c_val = (
                0.5 * inext_gaussian_A_nu_val * nu_val
                * rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c_val = (
                0.5 * inext_gaussian_A_nu_val * nu_val
                * rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val
            )

            rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit_val = (
                zeta_nu_char / 4.
            )
            rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit_val = (
                nu_val
                * rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_quarter_zeta_nu_char_g_c_val = (
                0.5 * A_nu_val * nu_val
                * rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val = (
                0.25
            )
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit_val = (
                nu_val
                * rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_quarter_zeta_nu_char_overline_g_c_val = (
                0.5 * A_nu_val * nu_val
                * rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val
            )

            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c_val = (
                0.5 * inext_gaussian_A_nu_val * nu_val
                * rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit_val
            )
            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c_val = (
                0.5 * inext_gaussian_A_nu_val * nu_val
                * rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val
            )

            for f_c_max_val in beyer_2000_f_c_max_tau_b_list:
                xi_c_max_val = f_c_max_val * beta * l_nu_eq
                lmbda_nu_xi_c_max_val = (
                    rate_independent_AFM_exprmt_single_chain.lmbda_nu_xi_c_hat_func(
                        xi_c_max_val)
                )
                epsilon_cnu_sci_hat_crit_val = (
                    rate_independent_AFM_exprmt_single_chain.epsilon_cnu_sci_hat_func(
                        lmbda_nu_xi_c_max_val)
                )
                epsilon_c_sci_hat_crit_val = (
                    nu_val * epsilon_cnu_sci_hat_crit_val
                )
                g_c_val = 0.5 * A_nu_val * nu_val * epsilon_cnu_sci_hat_crit_val
                overline_epsilon_cnu_sci_hat_crit_val = (
                    epsilon_cnu_sci_hat_crit_val / zeta_nu_char
                )
                overline_epsilon_c_sci_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_sci_hat_crit_val
                )
                overline_g_c_val = (
                    0.5 * A_nu_val * nu_val
                    * overline_epsilon_cnu_sci_hat_crit_val
                )

                inext_gaussian_g_c_val = (
                    0.5 * inext_gaussian_A_nu_val * nu_val
                    * epsilon_cnu_sci_hat_crit_val
                )
                inext_gaussian_overline_g_c_val = (
                    0.5 * inext_gaussian_A_nu_val * nu_val
                    * overline_epsilon_cnu_sci_hat_crit_val
                )

                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit.append(
                    epsilon_cnu_sci_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit.append(
                    epsilon_c_sci_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_g_c.append(
                    g_c_val)
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit.append(
                    overline_epsilon_cnu_sci_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit.append(
                    overline_epsilon_c_sci_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c.append(
                    overline_g_c_val)
                
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c.append(
                    inext_gaussian_g_c_val)
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c.append(
                    inext_gaussian_overline_g_c_val)

            A_nu___nu_chunk_list_mpi_scatter.append(A_nu_val)
            inext_gaussian_A_nu___nu_chunk_list_mpi_scatter.append(
                inext_gaussian_A_nu_val)
            inext_gaussian_A_nu_err___nu_chunk_list_mpi_scatter.append(
                inext_gaussian_A_nu_err_val)

            rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_epsilon_cnu_diss_hat_crit_val)
            rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_epsilon_c_diss_hat_crit_val)
            rate_independent_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_g_c_val)
            rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_overline_epsilon_cnu_diss_hat_crit_val)
            rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_overline_epsilon_c_diss_hat_crit_val)
            rate_independent_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_overline_g_c_val)

            rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit_val)
            rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit_val)
            rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_g_c_val)
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val)
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit_val)
            rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_overline_g_c_val)

            rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c_val)
            rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c_val)

            rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit_val)
            rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit_val)
            rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_g_c_val)
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit_val)
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit_val)
            rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_overline_g_c_val)

            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c_val)
            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c_val)

            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_g_c)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c)
            
            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c)
            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c)
        
        A_nu___nu_chunk_list_mpi_split = self.comm.gather(
            A_nu___nu_chunk_list_mpi_scatter, root=0)
        inext_gaussian_A_nu___nu_chunk_list_mpi_split = self.comm.gather(
            inext_gaussian_A_nu___nu_chunk_list_mpi_scatter, root=0)
        inext_gaussian_A_nu_err___nu_chunk_list_mpi_split = self.comm.gather(
            inext_gaussian_A_nu_err___nu_chunk_list_mpi_scatter, root=0)
        
        rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_g_c___nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_g_c___nu_chunk_list_mpi_scatter, root=0)
        rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        self.comm.Barrier()
        
        if self.comm_rank == 0:
            print("Post-processing rate-independent calculations")
            
            A_nu___nu_chunk_list = []
            inext_gaussian_A_nu___nu_chunk_list = []
            inext_gaussian_A_nu_err___nu_chunk_list = []
            
            rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list = []
            rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list = []
            rate_independent_g_c___nu_chunk_list = []
            rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list = []
            rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list = []
            rate_independent_overline_g_c___nu_chunk_list = []
            
            rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list = []
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list = []
            
            rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list = []
            rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list = []

            rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list = []
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list = []
            rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list = []
            
            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list = []
            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list = []
            
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list = []

            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list = []

            for proc_indx in range(self.comm_size):
                for nu_chunk_indx in range(cp.nu_num_list_mpi_split[proc_indx]):
                    A_nu___nu_chunk_val = (
                        A_nu___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    inext_gaussian_A_nu___nu_chunk_val = (
                        inext_gaussian_A_nu___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    inext_gaussian_A_nu_err___nu_chunk_val = (
                        inext_gaussian_A_nu_err___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_epsilon_c_diss_hat_crit___nu_chunk_val = (
                        rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_g_c___nu_chunk_val = (
                        rate_independent_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_val = (
                        rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_overline_g_c___nu_chunk_val = (
                        rate_independent_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_val = (
                        rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_val = (
                        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_val = (
                        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    A_nu___nu_chunk_list.append(A_nu___nu_chunk_val)
                    inext_gaussian_A_nu___nu_chunk_list.append(
                        inext_gaussian_A_nu___nu_chunk_val)
                    inext_gaussian_A_nu_err___nu_chunk_list.append(
                        inext_gaussian_A_nu_err___nu_chunk_val)

                    rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_val)
                    rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list.append(
                        rate_independent_epsilon_c_diss_hat_crit___nu_chunk_val)
                    rate_independent_g_c___nu_chunk_list.append(
                        rate_independent_g_c___nu_chunk_val)
                    rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val)
                    rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list.append(
                        rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_val)
                    rate_independent_overline_g_c___nu_chunk_list.append(
                        rate_independent_overline_g_c___nu_chunk_val)

                    rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_val)
                    rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_val)

                    rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_val)
                    rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list.append(
                        rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_val)
                    
                    rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_val)
                    rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_val)
                    rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_val)

                    rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_val)
                    rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list.append(
                        rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_val)

                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_val)
                    
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list.append(
                        rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_val)
            
            save_pickle_object(
                self.savedir, A_nu___nu_chunk_list,
                data_file_prefix+"-A_nu___nu_chunk_list")
            save_pickle_object(
                self.savedir, inext_gaussian_A_nu___nu_chunk_list,
                data_file_prefix+"-inext_gaussian_A_nu___nu_chunk_list")
            save_pickle_object(
                self.savedir, inext_gaussian_A_nu_err___nu_chunk_list,
                data_file_prefix+"-inext_gaussian_A_nu_err___nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_overline_g_c___nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list")
        
        self.comm.Barrier()
        

        # Rate-dependent calculations

        if self.comm_rank == 0:
            print("Rate-dependent calculations")

        if self.comm_rank == 0:
            print("Rate-dependent force-controlled AFM experiments")
        
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_g_c = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c = []

            rate_dependent_single_chain = (
                RateDependentScissioncuFJC(nu=nu_val,
                                           zeta_nu_char=zeta_nu_char,
                                           kappa_nu=kappa_nu,
                                           omega_0=omega_0)
            )
            A_nu_val = rate_dependent_single_chain.A_nu
            f_c_crit_val = (
                rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
            ) # (nN*nm)/nm = nN
            f_c_steps = np.linspace(0, f_c_crit_val, cp.f_c_num_steps) # nN
            for f_c_dot_val in cp.f_c_dot_list:
                t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for f_c_indx in range(cp.f_c_num_steps):
                    t_val = t_steps[f_c_indx]
                    xi_c_val = (
                        f_c_steps[f_c_indx] * beta * l_nu_eq
                    ) # nN*nm/(nN*nm)
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

                    if f_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[f_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[f_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                epsilon_c_diss_hat_crit_val = (
                    nu_val * epsilon_cnu_diss_hat_crit_val
                )
                g_c_val = (
                    0.5 * A_nu_val * nu_val * epsilon_cnu_diss_hat_crit_val
                )
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char
                )
                overline_epsilon_c_diss_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_val = (
                    0.5 * A_nu_val * nu_val
                    * overline_epsilon_cnu_diss_hat_crit_val
                )
                
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit.append(
                    epsilon_c_diss_hat_crit_val)
                rate_dependent_frc_cntrld_AFM_exprmts_g_c.append(g_c_val)
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit.append(
                    overline_epsilon_c_diss_hat_crit_val)
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c.append(
                    overline_g_c_val)
            
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit)
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit)
            rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_g_c)
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit)
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit)
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c)
        
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        self.comm.Barrier()
        
        
        if self.comm_rank == 0:
            print("Rate-dependent displacement-controlled AFM experiments")
        
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_g_c = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c = []

            rate_dependent_single_chain = (
                RateDependentScissioncuFJC(nu=nu_val,
                                           zeta_nu_char=zeta_nu_char,
                                           kappa_nu=kappa_nu,
                                           omega_0=omega_0)
            )
            A_nu_val = rate_dependent_single_chain.A_nu
            lmbda_c_eq_crit_val = rate_dependent_single_chain.lmbda_c_eq_crit
            lmbda_c_eq_steps = (
                np.linspace(0, lmbda_c_eq_crit_val, cp.r_nu_num_steps)
            )
            for r_nu_dot_val in cp.r_nu_dot_list:
                lmbda_c_eq_dot_val = (
                    r_nu_dot_val / (nu_val*l_nu_eq)
                ) # (nm/sec)/nm = 1/sec
                t_steps = (
                    lmbda_c_eq_steps / lmbda_c_eq_dot_val
                ) # 1/(1/sec) = sec

                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for r_nu_indx in range(cp.r_nu_num_steps):
                    t_val = t_steps[r_nu_indx]
                    lmbda_c_eq_val = lmbda_c_eq_steps[r_nu_indx]
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

                    if r_nu_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[r_nu_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[r_nu_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                epsilon_c_diss_hat_crit_val = (
                    nu_val * epsilon_cnu_diss_hat_crit_val
                )
                g_c_val = (
                    0.5 * A_nu_val * nu_val * epsilon_cnu_diss_hat_crit_val
                )
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char
                )
                overline_epsilon_c_diss_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_val = (
                    0.5 * A_nu_val * nu_val
                    * overline_epsilon_cnu_diss_hat_crit_val
                )
                
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val)
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit.append(
                    epsilon_c_diss_hat_crit_val)
                rate_dependent_strn_cntrld_AFM_exprmts_g_c.append(g_c_val)
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val)
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit.append(
                    overline_epsilon_c_diss_hat_crit_val)
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c.append(
                    overline_g_c_val)
            
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit)
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit)
            rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_g_c)
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit)
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit)
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c)
        
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_scatter,
                root=0)
        )
        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_split = (
            self.comm.gather(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_scatter,
                root=0)
        )
        
        self.comm.Barrier()

        if self.comm_rank == 0:
            print("Post-processing rate-dependent calculations")

            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list = []

            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list = []

            for proc_indx in range(self.comm_size):
                for nu_chunk_indx in range(cp.nu_num_list_mpi_split[proc_indx]):
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_val = (
                        rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_val = (
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_val = (
                        rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_val = (
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_val = (
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    )

                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list.append(
                        rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list.append(
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_val)

                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list.append(
                        rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list.append(
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list.append(
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_val)
            
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list")

    def finalization(self):
        """Define finalization analysis"""

        if self.comm_rank == 0:
            print(self.paper_authors+" "+self.chain+" finalization")

        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        polymer_type = cp.paper_authors2polymer_type_dict[self.paper_authors]
        chain_backbone_bond_type = (
            cp.polymer_type_label2chain_backbone_bond_type_dict[polymer_type]
        )
        data_file_prefix = (
            self.paper_authors + '-' + polymer_type + '-'
            + chain_backbone_bond_type + '-' + self.chain
        )
        
        zeta_nu_char = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt')
        
        if self.comm_rank == 0:
            print("Plotting")

            A_nu___nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-A_nu___nu_chunk_list")
            inext_gaussian_A_nu___nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-inext_gaussian_A_nu___nu_chunk_list")
            inext_gaussian_A_nu_err___nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-inext_gaussian_A_nu_err___nu_chunk_list")
            
            rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            )
            rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list")
            )
            rate_independent_g_c___nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_g_c___nu_chunk_list")
            rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            )
            rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list")
            )
            rate_independent_overline_g_c___nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_overline_g_c___nu_chunk_list")
            
            rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list")
            )
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list")
            )

            rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list")
            )
            rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list")
            )

            rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list")
            )
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list")
            )

            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list")
            )
            rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list")
            )

            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list")
            )
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list")
            )
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list")
            )

            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list")
            )
            rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list")
            )

            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list")
            )
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list")
            )

            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list")
            )
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list")
            )
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list")
            )

            # plot results
            latex_formatting_figure(ppp)

            fig, (ax1, ax2) = plt.subplots(
                2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
            
            ax1.semilogx(
                cp.nu_list, A_nu___nu_chunk_list,
                linestyle=cp.cufjc_A_nu_linestyle,
                color='blue', alpha=1, linewidth=2.5,
                label=r'$\textrm{c}u\textrm{FJC scission}$')
            ax1.semilogx(
                cp.nu_list, inext_gaussian_A_nu___nu_chunk_list,
                linestyle=cp.inext_gaussian_A_nu_linestyle,
                color='red', alpha=1, linewidth=2.5,
                label=r'$\textrm{inextensible Gaussian chain}$')
            ax1.legend(loc='best', fontsize=14)
            ax1.tick_params(axis='y', labelsize=20)
            ax1.set_ylabel(r'$\mathcal{A}_{\nu}$', fontsize=30)
            ax1.grid(True, alpha=0.25)
            
            ax2.loglog(
                cp.nu_list, inext_gaussian_A_nu_err___nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5)
            ax2.tick_params(axis='y', labelsize=20)
            ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=30)
            ax2.grid(True, alpha=0.25)
            
            plt.xticks(fontsize=20)
            plt.xlabel(r'$\nu$', fontsize=30)
            save_current_figure_no_labels(
                self.savedir,
                data_file_prefix+"-A_nu-cufjc-igc-comparison")

            fig = plt.figure()
            fig_legend = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(
                [], [], linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            ax.plot(
                [], [], linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                ax.plot(
                    [], [],
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            ax.plot(
                [], [],
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                ax.plot(
                    [], [],
                    linestyle=cp.f_c_dot_linestyle_list[f_c_dot_indx],
                    color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            fig_legend.legend(
                ax.get_legend_handles_labels()[0],
                ax.get_legend_handles_labels()[1],
                loc='center', fontsize=20)
            fig_legend.savefig(
                self.savedir+data_file_prefix+"-force-control-main-text-legend"+".pdf",
                transparent=True)
            plt.close()

            fig = plt.figure()
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list,
                rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.f_c_dot_linestyle_list[f_c_dot_indx],
                    color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$(\hat{\varepsilon}_{c\nu}^{sci})^{crit},~(\hat{\varepsilon}_{c\nu}^{diss})^{crit}$', 30,
                data_file_prefix+"-force-control-nondim-crit-diss-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_overline_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list,
                rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_overline_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.f_c_dot_linestyle_list[f_c_dot_indx],
                    color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_overline_label_list[f_c_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{sci})^{crit}},~\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$', 30,
                data_file_prefix+"-force-control-nondim-scaled-crit-diss-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list,
                rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.f_c_dot_linestyle_list[f_c_dot_indx],
                    color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$(\hat{\varepsilon}_c^{sci})^{crit},~(\hat{\varepsilon}_c^{diss})^{crit}$', 30,
                data_file_prefix+"-force-control-nondim-crit-diss-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_overline_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list,
                rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_overline_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.f_c_dot_linestyle_list[f_c_dot_indx],
                    color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_overline_label_list[f_c_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$\overline{(\hat{\varepsilon}_c^{sci})^{crit}},~\overline{(\hat{\varepsilon}_c^{diss})^{crit}}$', 30,
                data_file_prefix+"-force-control-nondim-scaled-crit-diss-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c___nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_list,
                    linestyle=cp.f_c_dot_linestyle_list[f_c_dot_indx],
                    color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$g_c$', 30,
                data_file_prefix+"-force-control-nondim-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_overline_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_overline_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_list,
                    linestyle=cp.f_c_dot_linestyle_list[f_c_dot_indx],
                    color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_overline_label_list[f_c_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$\overline{g_c}$', 30,
                data_file_prefix+"-force-control-nondim-scaled-fracture-toughness-vs-nu")

            fig = plt.figure()
            fig_legend = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(
                [], [], linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            ax.plot(
                [], [], linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                ax.plot(
                    [], [],
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            ax.plot(
                [], [],
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                ax.plot(
                    [], [],
                    linestyle=cp.r_nu_dot_linestyle_list[r_nu_dot_indx],
                    color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            fig_legend.legend(
                ax.get_legend_handles_labels()[0],
                ax.get_legend_handles_labels()[1],
                loc='center', fontsize=20)
            fig_legend.savefig(
                self.savedir+data_file_prefix+"-displacement-control-main-text-legend"+".pdf",
                transparent=True)
            plt.close()
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list,
                rate_independent_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.r_nu_dot_linestyle_list[r_nu_dot_indx],
                    color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$(\hat{\varepsilon}_{c\nu}^{sci})^{crit},~(\hat{\varepsilon}_{c\nu}^{diss})^{crit}$', 30,
                data_file_prefix+"-displacement-control-nondim-crit-diss-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_overline_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list, rate_independent_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_overline_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.r_nu_dot_linestyle_list[r_nu_dot_indx],
                    color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_overline_label_list[r_nu_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{sci})^{crit}},~\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$', 30,
                data_file_prefix+"-displacement-control-nondim-scaled-crit-diss-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list,
                rate_independent_epsilon_c_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit___nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.r_nu_dot_linestyle_list[r_nu_dot_indx],
                    color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$(\hat{\varepsilon}_c^{sci})^{crit},~(\hat{\varepsilon}_c^{diss})^{crit}$', 30,
                data_file_prefix+"-displacement-control-nondim-crit-diss-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_full_zeta_nu_char_label)
            plt.semilogx(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_overline_epsilon_c_sci_hat_crit___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_sci_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_overline_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list,
                rate_independent_overline_epsilon_c_diss_hat_crit___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_overline_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit___nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.r_nu_dot_linestyle_list[r_nu_dot_indx],
                    color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_overline_label_list[r_nu_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$\overline{(\hat{\varepsilon}_c^{sci})^{crit}},~\overline{(\hat{\varepsilon}_c^{diss})^{crit}}$', 30,
                data_file_prefix+"-displacement-control-nondim-scaled-crit-diss-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c___nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_list,
                    linestyle=cp.r_nu_dot_linestyle_list[r_nu_dot_indx],
                    color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$g_c$', 30,
                data_file_prefix+"-displacement-control-nondim-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                linestyle=cp.LT_full_zeta_nu_char_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                linestyle=cp.LT_quarter_zeta_nu_char_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_overline_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c___nu_chunk_list,
                linestyle=cp.cufjc_linestyle, color=cp.cufjc_color,
                alpha=1, linewidth=2.5, label=cp.cufjc_overline_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c___nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_list,
                    linestyle=cp.r_nu_dot_linestyle_list[r_nu_dot_indx],
                    color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_overline_label_list[r_nu_dot_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$\overline{g_c}$', 30,
                data_file_prefix+"-displacement-control-nondim-scaled-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            if cp.f_c_dot_num == cp.r_nu_dot_num:
                for indx in range(cp.f_c_dot_num):
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list = [
                        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][indx]
                        for nu_chunk_indx in range(cp.nu_num)
                    ]
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list = [
                        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][indx]
                        for nu_chunk_indx in range(cp.nu_num)
                    ]
                    plt.semilogx(
                        cp.nu_list,
                        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list,
                        linestyle=cp.f_c_dot_comparison_linestyle_list[indx],
                        color=cp.f_c_dot_comparison_color_list[indx],
                        alpha=1, linewidth=2.5,
                        label=cp.f_c_dot_comparison_label_list[indx])
                    plt.semilogx(
                        cp.nu_list,
                        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list,
                        linestyle=cp.r_nu_dot_comparison_linestyle_list[indx],
                        color=cp.r_nu_dot_comparison_color_list[indx],
                        alpha=1, linewidth=2.5,
                        label=cp.r_nu_dot_comparison_label_list[indx])
            plt.legend(loc='best', fontsize=12)
            plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$(\hat{\varepsilon}_{c\nu}^{diss})^{crit}$', 30,
                data_file_prefix+"-control-comparison-nondim-crit-diss-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            if cp.f_c_dot_num == cp.r_nu_dot_num:
                for indx in range(cp.f_c_dot_num):
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list = [
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][indx]
                        for nu_chunk_indx in range(cp.nu_num)
                    ]
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list = [
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit___nu_chunk_list[nu_chunk_indx][indx]
                        for nu_chunk_indx in range(cp.nu_num)
                    ]
                    plt.semilogx(
                        cp.nu_list,
                        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list,
                        linestyle=cp.f_c_dot_comparison_linestyle_list[indx],
                        color=cp.f_c_dot_comparison_color_list[indx],
                        alpha=1, linewidth=2.5,
                        label=cp.f_c_dot_comparison_overline_label_list[indx])
                    plt.semilogx(
                        cp.nu_list,
                        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list,
                        linestyle=cp.r_nu_dot_comparison_linestyle_list[indx],
                        color=cp.r_nu_dot_comparison_color_list[indx],
                        alpha=1, linewidth=2.5,
                        label=cp.r_nu_dot_comparison_overline_label_list[indx])
            plt.legend(loc='best', fontsize=12)
            plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30,
                r'$\overline{(\hat{\varepsilon}_{c\nu}^{diss})^{crit}}$', 30,
                data_file_prefix+"-control-comparison-nondim-scaled-crit-diss-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            fig_legend = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_full_zeta_nu_char_label)
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_overline_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                ax.plot(
                    [], [],
                    linestyle=cp.cufjc_A_nu_linestyle,
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_overline_label_list[AFM_expermts_indx])
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color='white', alpha=1, linewidth=2.5,
                label=' ')
            ax.plot(
                [], [], linestyle=cp.inext_gaussian_A_nu_linestyle,
                color='black', alpha=1, linewidth=2.5,
                label=r'$\textrm{inextensible Gaussian chain}~\mathcal{A}_{\nu}=1/\sqrt{\nu}$')
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color='black', alpha=1, linewidth=2.5,
                label=r'$\textrm{c}u\textrm{FJC scission}~\mathcal{A}_{\nu}$')
            fig_legend.legend(
                ax.get_legend_handles_labels()[0],
                ax.get_legend_handles_labels()[1],
                loc='center', fontsize=20)
            fig_legend.savefig(
                self.savedir+data_file_prefix+"-A_nu-cufjc-igc-comparison-nondim-scaled-supporting-information-legend"+".pdf",
                transparent=True)
            plt.close()

            fig = plt.figure()
            fig_legend = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                ax.plot(
                    [], [],
                    linestyle=cp.cufjc_A_nu_linestyle,
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color='white', alpha=1, linewidth=2.5,
                label=' ')
            ax.plot(
                [], [], linestyle=cp.inext_gaussian_A_nu_linestyle,
                color='black', alpha=1, linewidth=2.5,
                label=r'$\textrm{inextensible Gaussian chain}~\mathcal{A}_{\nu}=1/\sqrt{\nu}$')
            ax.plot(
                [], [], linestyle=cp.cufjc_A_nu_linestyle,
                color='black', alpha=1, linewidth=2.5,
                label=r'$\textrm{c}u\textrm{FJC scission}~\mathcal{A}_{\nu}$')
            fig_legend.legend(
                ax.get_legend_handles_labels()[0],
                ax.get_legend_handles_labels()[1],
                loc='center', fontsize=20)
            fig_legend.savefig(
                self.savedir+data_file_prefix+"-A_nu-cufjc-igc-comparison-nondim-supporting-information-legend"+".pdf",
                transparent=True)
            plt.close()
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                linestyle=cp.inext_gaussian_A_nu_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_g_c___nu_chunk_list,
                linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_g_c___nu_chunk_list,
                linestyle=cp.inext_gaussian_A_nu_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_g_c___nu_chunk_list,
                linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_g_c_list,
                    linestyle=cp.inext_gaussian_A_nu_linestyle,
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_list,
                    linestyle=cp.cufjc_A_nu_linestyle,
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$g_c$', 30,
                data_file_prefix+"-A_nu-cufjc-igc-comparison-nondim-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                linestyle=cp.inext_gaussian_A_nu_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_full_zeta_nu_char_overline_g_c___nu_chunk_list,
                linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_full_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_full_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_inext_gaussian_overline_g_c___nu_chunk_list,
                linestyle=cp.inext_gaussian_A_nu_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            plt.loglog(
                cp.nu_list,
                rate_independent_LT_quarter_zeta_nu_char_overline_g_c___nu_chunk_list,
                linestyle=cp.cufjc_A_nu_linestyle,
                color=cp.LT_quarter_zeta_nu_char_color, alpha=1, linewidth=2.5,
                label=cp.LT_quarter_zeta_nu_char_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c___nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_inext_gaussian_overline_g_c_list,
                    linestyle=cp.inext_gaussian_A_nu_linestyle,
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_list,
                    linestyle=cp.cufjc_A_nu_linestyle,
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            # plt.legend(loc='best', fontsize=12)
            # plt.ylim([-0.015, 1.015])
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 30, r'$\overline{g_c}$', 30,
                data_file_prefix+"-A_nu-cufjc-igc-comparison-nondim-scaled-fracture-toughness-vs-nu")


if __name__ == '__main__':

    start_time = time.process_time()

    T = 298 # absolute room temperature, K

    AFM_chain_tensile_tests_dict = {
        "al-maawali-et-al": "chain-a", "hugel-et-al": "chain-a"
    }

    al_maawali_et_al_fracture_toughness_characterizer = (
        FractureToughnessCharacterizer(
            paper_authors="al-maawali-et-al", chain="chain-a", T=T)
    )
    al_maawali_et_al_fracture_toughness_characterizer.characterization()
    # al_maawali_et_al_fracture_toughness_characterizer.finalization()

    del al_maawali_et_al_fracture_toughness_characterizer

    hugel_et_al_fracture_toughness_characterizer = (
        FractureToughnessCharacterizer(
            paper_authors="hugel-et-al", chain="chain-a", T=T)
    )
    hugel_et_al_fracture_toughness_characterizer.characterization()
    # hugel_et_al_fracture_toughness_characterizer.finalization()

    del hugel_et_al_fracture_toughness_characterizer

    elapsed_wall_time = time.process_time() - start_time
    print(elapsed_wall_time)