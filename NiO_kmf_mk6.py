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

filename = os.path.basename(__file__)  #gets file name
fileroot = os.path.splitext(os.path.basename(filename))[0] #removes .py from filename
current_directory = os.getcwd()

c_len = 4.17
cell = gto.Cell()
cell.unit = 'Angstrom' #Is already this by default, added for clarity
cell.atom =f''' Ni  0.          0.          0.
                Ni  {c_len}     {c_len}     {c_len} 
                O   {c_len/2}   {c_len/2}   {c_len/2}
                O   {3*c_len/2} {3*c_len/2} {3*c_len/2}'''
cell.basis = {'Ni': 'gth-dzvp-molopt-sr', 'O': 'gth-dzvp-molopt-sr'}
cell.pseudo = {'Ni': 'gth-pbe-q18','O': 'gth-pbe-q6'}
cell.a = f'''       {c_len}     {c_len/2}   {c_len/2}
                    {c_len/2}   {c_len}     {c_len/2}
                    {c_len/2}   {c_len/2}   {c_len}'''
#double primitive cell so magnetization can be determined                    
cell.verbose = 5
cell.precision = 1e-8
cell.exp_to_discard = 0.1
cell.build()



T = 0.01    #see notes in KMF section about sigma and smearing
k = 2
kmesh = [k,k,k]
Lat = lattice.Lattice(cell,kmesh)
kpts = Lat.kpts

#sets names for chkfile for k and k-1
fn_k = f"{fileroot}_k{k}.chk"
fn_kmi1 = f"{fileroot}_k{k-1}.chk"
#searches for k and k-1 chkfiles
chk_k = os.path.isfile(os.path.join(current_directory, fn_k))
chk_kmi1 = os.path.isfile(os.path.join(current_directory, fn_kmi1))

# Gaussian Density Fitting
exxdiv = None  #'ewald'
gdf_fname = f"gdf_ints_{fileroot}_k{k}.h5" #add {os.getpid()} to run multiple in parallel
#{fileroot} and _k{k} added to gdf for k-specific calling schema
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.build()
gdf.force_dm_kbuild = True

# K Point Mean-Field
kmf= scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
kmf.max_cycle=200
kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-8

#logic for kmf.init_guess based on chkfile existence
if chk_k is True:               #looks for chkfile for same script and same k value
    chkfile = fn_k              
    kmf.init_guess = 'chkfile'  #if present, sets initial guess to that chkfile
elif chk_kmi1 is True:          #if chkfile for same script and k value aren't present, looks for chkfile for script and k-1
    chkfile = fn_kmi1
    kmf.init_guess = 'chkfile'  #if chkfile for script and k-1 are present, sets initial guess to that value
else:                           #if no applicable chkfiles exist, sets the initial guess otherwise
    kmf.init_guess = 'vsap'               #'minao' as other option
    print("No chkfile available.")

kmf = pbc_hp.smearing_(kmf, sigma = T)      #smaller sigma -> more accurate but slower 
kmf.chkfile = f"{fileroot}_k{k}.chk"        #writes to a chkfile named after file and k value
kmf.kernel()

e_tot_mf    = kmf.e_tot         #total mean-field energy (electronic + nuclear)
e_nuc       = cell.energy_nuc() #nuclear energy from PySCF
e_elec_mf = e_tot_mf -  e_nuc   #electronic energy

if chk_k is True:
    print(f"This system's k = {k}, chkfile was for k={k}.")
elif chk_kmi1 is True:
    print(f"This system's k = {k}, chkfile was for k={k-1}.")
else:
    print("This system's k = {k}, no chkfile used.")

print(f"{c_len} {e_elec_mf} {e_nuc} {e_tot_mf}")
exit()