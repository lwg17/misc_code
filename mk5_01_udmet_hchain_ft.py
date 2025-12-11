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

T = 0.01 # finite temperature

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


dm = kmf.make_rdm1()                            #make rdm1 matrix
S = np.asarray(kmf.get_ovlp())
if S.ndim == 2:
    S = S[None, :, :]
nk = S.shape[0]



wk = getattr(kmf, "kpts_weights", None)         #get k point weights
if wk is None:                                  #uniform weight if no k point weight
        wk = np.ones(nk)/nk                     
        
aoslice = cell.aoslice_by_atom()                #build aoslices
natm = cell.natm                                # counts number of atoms
Nalpha = np.zeros(natm)                         #array of atoms for Nalpha
Nbeta  = np.zeros(natm)                         #array of atoms for Nbeta

for k in range(nk):                             #loop over all k points
        Sk = S[k]                               #selects overlap matrix at k point
        Da = dm[0,k]                            #selects alpha density matrix at k point
        Db = dm[1,k]                            #selects beta density matrix at k point

        pa = np.real(np.diag(Da @ Sk))                  #Compute per AO population of diagonal of DS
        pb = np.real(np.diag(Db @ Sk))                  #Compute per AO population of diagonal of DS

        for A in range(natm):                           #Sums AO population for each atom weight-averegd over k
                p0, p1 = aoslice[A, 2], aoslice[A, 3]
                Nalpha[A] += wk[k] * pa[p0:p1].sum()
                Nbeta[A]  += wk[k] * pb[p0:p1].sum()
        
m = Nalpha - Nbeta                              #builds Magnetic moment
M = float(m.sum())                              #net Moment
abs_m = float(np.abs(m).sum())                  #total moment scale

eps = 1e-3                                      #tolerance for zero moment
if abs_m < eps:                                 #conditionals to label magnetism
        label = "Paramagnetic"
elif abs(M) > 0.5 * abs_m:
        label = "Ferromagnetic"
elif abs(M) < 0.1 * abs_m and natm>=2 and np.sign(m[0]) ==-np.sign(m[1]):
        label = "AntiFerromagnetic"
else:
        label = "mixed/complex"

print("\n=== Per-atom Mulliken spin populations (k-averaged) ===")
for A in range(natm):
    sym = cell.atom_symbol(A)
    print(f"Atom {A} ({sym}): Nα={Nalpha[A]: .12f}  Nβ={Nbeta[A]: .12f}  m={m[A]: .12f}")
print(f"Total electrons (Σ(Nα+Nβ)) = {float(np.sum(Nalpha+Nbeta)):.12f}")
print(f"Net magnetization M = Σ m_A = {M:.12f}")
print(f"Spin order label = {label}")
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
