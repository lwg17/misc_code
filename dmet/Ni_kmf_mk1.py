import os
import numpy as np
from pyscf.pbc import gto, scf, df
from libdmet.system import lattice
from libdmet.lo import make_lo
from libdmet.dmet import udmet
from libdmet.solver import cc_solver, fci_solver
from libdmet.solver.ft_solver import fted_solver, ftdmrg_solver, ltdmrg_solver
from libdmet.mean_field import pbc_helper as pbc_hp
from libdmet.mean_field import mfd

np.set_printoptions(4, linewidth=1000, suppress=True)


c_len=3.48
cell = gto.Cell()

cell.atom =f''' Ni  0.          0.          0.
                Ni  {1/2*c_len} {1/2*c_len} 0.
                Ni  {1/2*c_len} 0.          {1/2*c_len}
                Ni  0           {1/2*c_len} {1/2*c_len}'''
cell.basis = {'Ni': 'gth-dzvp-molopt-sr'}
cell.pseudo = {'Ni': 'gth-pbe-q18'}
cell.a = np.eye(3)*c_len
cell.verbose = 5
cell.precision = 1e-8
cell.exp_to_discard = 0.1
cell.build()

k = 1
kmesh = [k,k,k]
Lat = lattice.Lattice(cell,kmesh)
kpts = Lat.kpts


# Gaussian Density Fitting
exxdiv = None
gdf_fname = f"gdf_ints_{os.getpid()}.h5" #added {os.getpid()} so can run multiple in parallel
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.build()
gdf.force_dm_kbuild = True

# K Point Mean-Field
kmf= scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-8
#kmf = pbc_hp.smearing_(kmf, sigma = T)
kmf.kernel()

e_tot_mf    = kmf.e_tot         #total mean-field energy (electronic + nuclear)
e_nuc       = cell.energy_nuc() #nuclear energy from PySCF
e_elec_mf = e_tot_mf -  e_nuc   #electronic energy

print(f"{c_len} {e_elec_mf} {e_nuc} {e_tot_mf}")
exit()