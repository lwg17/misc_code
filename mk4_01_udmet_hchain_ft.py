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

T = 0.0 # finite temperature

# Set up the system
basis = '321g'
#R_b = float(os.environ.get("R_b", "1.00"))
R_b = 1.5
R=R_b * 0.529177249
cell = gto.Cell()
cell.unit = 'Angstrom' #Is already this by default, added for clarity
cell.a    = f"""5.0    0.0    0.0
                0.0    5.0    0.0
                0.0    0.0    {2*R} """
cell.atom = f""" H 0.0    0.0    0.0
                H 0.0    0.0    {R} """
cell.basis = basis
cell.verbose = 5
cell.precision = 1e-12
cell.build(dump_input=False)

kmesh = [1, 1, 3]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts

# Density fitting
exxdiv = None
gdf_fname = f"gdf_ints_{os.getpid()}.h5" #added {os.getpid()} so can run multiple in parallel
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.build()
gdf.force_dm_kbuild = True

# mean-field solver

kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-12
kmf = pbc_hp.smearing_(kmf, sigma=T)
kmf.kernel()

#dm1_kpts = kmf.make_rdm1() #1rdm matrix
dm = kmf.make_rdm1()
S = kmf.get_ovlp()
nk = len(S)
wk = getattr(kmf, "kpts_weights",None)
if wk is None:
                wk = np.ones(nk)/nk
                
aoslice=cell.aoslice_by_atom()
Porth = np.zeros((cell.nao, cell.nao), dtype=complex)


for k in range(nk):
    # S^(1/2) via eigendecomposition (Sk is Hermitian)
    e, U = np.linalg.eigh(S[k])
    # guard tiny/negative eigenvalues from numerical noise
    e = np.clip(e, 0.0, None)
    S12 = (U * np.sqrt(e)) @ U.conj().T

    # spin-sum at this k
    Dk = dm[0, k] + dm[1, k]

    # Löwdin-orthonormal density matrix at this k
    Pk = S12 @ Dk @ S12

    Porth += wk[k] * Pk

Ne_total = np.trace(Porth).real
print("Orthonormal RDM-1:")
print(Porth)
print("Total electrons from Tr(Porth):", Ne_total)

for ia in range(cell.natm):
    p0, p1 = aoslice[ia, 2], aoslice[ia, 3]
    Ne_a = np.trace(Porth[p0:p1, p0:p1]).real
    print(f"Atom {ia} ({cell.atom_symbol(ia)}): electrons = {Ne_a}")



#ao_pop, charges = kmf.mulliken_meta()   #mulliken population analysis calc

#e_tot_mf = kmf.e_tot              # total MF energy (electronic + nuclear)
#e_nuc    = cell.energy_nuc()      # nuclear energy from PySCF
#e_elec_mf = e_tot_mf - e_nuc      # electronic-only part
#for i, lbl in enumerate(cell.ao_labels()):
#    print(i, lbl)

#print("\n=== 1-RDM (alpha/beta, kpts, nao, nao) ===") #1rdm matrix header
#print(dm1_kpts)                                       #1rdm matrix

#print("\n=== Mulliken AO populations (meta-Löwdin) ===")
#print(ao_pop)
#print("\n=== Mulliken atomic charges ===")
#print(charges)
#print("\nTotal electrons from Mulliken AO populations:", np.sum(ao_pop))
#print(f"{R_b} {e_elec_mf} {e_nuc} {e_tot_mf}")

#from interpret_rdm1_kpts import interpret_from_kmf
#interpret_from_kmf(kmf, cell, top_occ=10)
exit()


# define local orbitals
C_ao_lo, idx_core, idx_val, idx_virt = \
        make_lo.get_iao(kmf, minao='scf', full_return=True)


# define impurity information
val_idx, virt_idx = Lat.build(idx_val=idx_val, idx_virt=idx_virt)

# define solver
if T < 0.01:
        beta = 1000
else:
        beta = 1./T
if beta > 100:
        solver = cc_solver.CCSolver(restricted=False, restart=True, verbose=cell.verbose, beta=beta)
        solver_argss = [["C_lo_eo", ("fname_cc", "fucc.h5")]]
# 
# solver = fci_solver.FCISolver(restricted=False)
else:
        solver = fted_solver.FTEDSolver(restricted=False, beta=beta, solve_mu=True)
        # solver = ftdmrg_solver.FTDMRGSolver(beta=beta, restricted=False, verbose=0)
        # solver = ltdmrg_solver.LTDMRGSolver(beta=beta, mu_gc=0, restricted=False, verbose=0)
        solver_argss = None
# solve dmet
mydmet = udmet.UDMET(Lat, kmf, solver, C_ao_lo, vcor=None,
                        solver_argss=solver_argss)
mydmet.dump_flags()

# get the global Mu for finite temperature 
res = mydmet.solve_mf(return_mu=True)
ft_mu = res[-1]

#mydmet.calc_e_tot = False
#mydmet.max_cycle = 5
mydmet.mu_glob = np.array([0.2, -0.1])
mydmet.beta = beta
mydmet.bath_power = 1
mydmet.fit_method = 'CG'
mydmet.fit_kwargs = {"test_grad": True}
mydmet.kernel()

e_tot_mf = kmf.e_tot              # total MF energy (electronic + nuclear)
e_nuc    = cell.energy_nuc()      # nuclear energy from PySCF
e_elec_mf = e_tot_mf - e_nuc      # electronic-only part
print(f"{R_b} {e_elec_mf} {e_nuc} {e_tot_mf}")


#e_nuc        = mydmet.h0              # same as KMF nuclear
#e_tot_dmet   = mydmet.e_tot           # DMET total (elec + h0)
#e_elec_dmet  = e_tot_dmet - e_nuc     # DMET electronic-only
#print(f"{R_b} {e_elec_dmet} {e_nuc} {e_tot_dmet}")

#print(e_mf)

#print ("resulting rdm1")
#print (mydmet.rdm1_glob[:, 0])
#print(f"DMET energy: {mydmet.e_tot}")
# compare to rdmet
#print ("energy diff to ref", abs(mydmet.e_tot - -0.8605063783))
# assert abs(mydmet.e_tot - -0.8605063783) < 1e-5
