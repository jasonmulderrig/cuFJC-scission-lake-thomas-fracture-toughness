#############################################
cuFJC scission Lake-Thomas fracture toughness
#############################################

A repository that incorporates the composite uFJC (cuFJC) model with scission in the Lake-Thomas theory of polymer fracture. This repository is dependent upon the `cuFJC-scission <https://pypi.org/project/cufjc-scission/>`_ Python packages.

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Python virtual environment associated with the project needs to be installed. The installation of this Python virtual environment and some essential packages is handled by the ``virtual-environment-install-master.sh`` Bash script. Before running the script, make sure to change the ``VENV_PATH`` parameter to comply with your personal filetree structure.

*****
Usage
*****

The Python codes in this repository are organized within separate directories. The Python codes in each directory correspond to a particular theoretical modeling variation of the Lake-Thomas theory of polymer fracture. Note that each directory contains three essential back-end Python codes that provide miscellaneous support: ``characterizer.py``, ``default_parameters.py``, and ``utility.py``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Crack plane bridging chains in a uniform network -- ``uniform-network-bridging-chains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are four working Python codes in this directory: ``AFM_chain_tensile_test_curve_fit.py``, ``fracture_toughness.py``, ``fracture_toughness_comprehensive_sweep.py``, and ``fracture_toughness_primary_sweep.py``. Each of these codes creates an object that can perform a "characterization" routine, a "finalization" routine, or both routines (where the characterization routine must preceed the finalization routine). The characterization routine performs a variety of calculations, and saves the results of those calculations in pickle files or text files stored within appropriately named directories (``.\AFM_chain_tensile_test_curve_fit\``, ``.\fracture_toughness\``, ``.\fracture_toughness_comprehensive_sweep\``, and ``.\fracture_toughness_primary_sweep\``). The finalization routine then loads the contents of those pickle files or text files, plots the results, and saves the plots back in the appropriate directory. Before running each Python code, make sure you confirm which routines you want to execute. As a means of precaution, the codes provided in this repository only have the finalization routine activated (with the characterization routine commented out, but this routine can be un-commented out if desired).

These Python codes are written in a parallelized computing fashion, *vis-à-vis* the ``mpi4py`` package. In order to run each code, first activate the Python virtual environment, and then execute the following command in the terminal:

::

    mpiexec -n number_of_cores python3 {AFM_chain_tensile_test_curve_fit, fracture_toughness, fracture_toughness_comprehensive_sweep, fracture_toughness_primary_sweep}.py

You can modify the number of cores called upon by replacing ``number_of_cores`` with that number. In my experience, the runtime of the characterization routine seems to be optimized when calling upon 8 cores. On my computer running 8 cores, the characterization routine of the ``AFM_chain_tensile_test_curve_fit.py``, ``fracture_toughness.py``, ``fracture_toughness_comprehensive_sweep.py``, and ``fracture_toughness_primary_sweep.py`` codes takes roughly 1 minute, 1 hour, 1 hour, and 1.5-2 days to complete, respectively.

***********
Information
***********

- `Releases <https://github.com/jasonmulderrig/cuFJC-scission-lake-thomas-fracture-toughness/releases>`__
- `Repository <https://github.com/jasonmulderrig/cuFJC-scission-lake-thomas-fracture-toughness>`__

********
Citation
********

\Jason Mulderrig, Franck Vernerey, and Nikolaos Bouklas, Accounting for chain statistics and loading rate sensitivity in the Lake-Thomas theory of polymer fracture, In preparation.

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, ``cufjc-scission``: the Python package for the composite uFJC (cuFJC) model with scission, `Zenodo (2024) <https://doi.org/10.5281/zenodo.10879757>`_.

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, A statistical mechanics framework for polymer chain scission, based on the concepts of distorted bond potential and asymptotic matching, `Journal of the Mechanics and Physics of Solids 174, 105244 (2023) <https://www.sciencedirect.com/science/article/pii/S0022509623000480>`_.
