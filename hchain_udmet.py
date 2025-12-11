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

T = 0.01 # finite temperature

# Set up the system
basis = '321g'  #finite temperature
R_b = float(os.environ.get("R_b", "1.00"))
#R_b = 4
R=R_b * 0.529177249     #convert input R(bohr) to R(angstrom)
cell = gto.Cell()
cell.unit = 'Angstrom' #Is already this by default, added for clarity
cell.a    = f"""5.0    0.0    0.0
                0.0    5.0    0.0
                0.0    0.0    {2*R} """
cell.atom = f""" H 0.0    0.0    0.0
                H 0.0    0.0    {R} """
#establishes lattice shape and atom placement
cell.basis = basis
cell.verbose = 5
#higher verbosity => more detailed output
cell.precision = 1e-12
cell.build(dump_input=False)

k = 3
kmesh = [1, 1, k]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts

# Density fitting
exxdiv = None
gdf_fname = f"gdf_ints_{fileroot}.h5" #add {os.getpid()} to run multiple in parallel
#{fileroot} used so different scripts can run at once
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.build()
gdf.force_dm_kbuild = True

# mean-field solver

kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-12
#kmf = pbc_hp.smearing_(kmf, sigma=T)

dm0 = np.asarray(kmf.get_init_guess(cell, key="minao"))
#Builds initial density matrix using minao, converts to array
dm0[0][0,0] = 2         #perturbation of alpha spin density matrix
    
kmf.kernel(dm0=dm0)     #runs SCF calculations starting with perturbed density matrix



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
mydmet.mu_glob = np.array(ft_mu, dtype=float)
mydmet.beta = beta
mydmet.bath_power = 1
mydmet.fit_method = 'CG'
mydmet.fit_kwargs = {"test_grad": True}
mydmet.kernel()
#build density matrix from dmet calculations
dm = mydmet.rdm1_glob
#make overlap matrix
S  = np.asarray(kmf.get_ovlp())
if S.ndim == 2:
    S = S[None, :, :]
nk = S.shape[0]

wk = getattr(kmf, "kpts_weights", None)
if wk is None:
    wk = np.ones(nk)

aoslice = cell.aoslice_by_atom()
#gets atomic orbital index range for each atom for mapping
natm = cell.natm
#stores number of atoms
Nalpha = np.zeros(natm)
Nbeta  = np.zeros(natm)
#builds arrays to store alpha and beta spins for LPA per atom

for k in range(nk):
    Sk = S[k]
    Da = dm[0, k]   #alpha-spin density matrix at k point
    Db = dm[1, k]   #beta-spin density matrix at point k

    # Löwdin S^(1/2) for this k-point
    e, U = np.linalg.eigh(Sk)
    e = np.clip(e, 0.0, None)  # kill tiny negative eigenvalues from numerical noise
    S12 = (U * np.sqrt(e)) @ U.conj().T
    #constructs the S_k^1/2, Lowdin square root of overlap matrix

    # Löwdin-orthogonal populations: diag(S^(1/2) D S^(1/2))
    Pa = S12 @ Da @ S12
    Pb = S12 @ Db @ S12
    #forms alpha and beta spin Lowdin populatin matrices (orthogonalization of Da/Db)
    pa = np.real(np.diag(Pa))
    pb = np.real(np.diag(Pb))
    #extracts real part of the diagonal elements of population matrices

    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        #gets atomic orbital index range for each atom
        Nalpha[A] += wk[k] * pa[p0:p1].sum()
        Nbeta[A]  += wk[k] * pb[p0:p1].sum()
        #adds k-weighted alpha/beta population sum

m = Nalpha - Nbeta
#per atom spin magnetization

M = float(m.sum())
#total net magnetixation
abs_m = float(np.abs(m).sum())
#sum of absolute per-atom megnetization to guage ordering

eps = 1e-3
if abs_m < eps:
    label = "Paramagnetic"
    #no localized moments => paramagnetic
elif abs(M) > 0.5 * abs_m:
    label = "Ferromagnetic"
    #magnetic moments aligned => ferromagnetic
elif abs(M) < 0.1 * abs_m and natm >= 2 and np.sign(m[0]) == -np.sign(m[1]):
    label = "AntiFerromagnetic"
    #small/no net magnetic moments, atoms have opposite spins => antiferromagnetic
    #only built to work on a 2-atom system, only checks first two atoms
else:
    label = "mixed/complex"
    #catchall
    
print("\n=== Per-atom Löwdin spin populations (k-averaged) ===")
for A in range(natm):
    sym = cell.atom_symbol(A)
    print(f"Atom {A} ({sym}): Nα={Nalpha[A]: .12f}  Nβ={Nbeta[A]: .12f}  m={m[A]: .12f}")
    #loops over alla toms to show individual spin populations
print(f"Total electrons (Σ(Nα+Nβ)) = {float(np.sum(Nalpha+Nbeta)):.12f}")
print(f"Net magnetization M = Σ m_A = {M:.12f}")
print(f"Spin order label = {label}")

print(f"Distance apart = {R_b} bohr.")

exit()