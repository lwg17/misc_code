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
filename = os.path.basename(__file__)                       #gets file name
fileroot = os.path.splitext(os.path.basename(filename))[0]  #removes .py from filename
current_directory = os.getcwd()                             #finds pwd

T = 0.01 # finite temperature

# Set up the system
basis = 'sto6g'
#R_b = float(os.environ.get("R_b", "1.00"))
R_b = 4.0
R=R_b * 0.529177249
r = round(R,1)
cell = gto.Cell()
X = 30
Y = 30

cell.unit = 'Angstrom' #Is already this by default, added for clarity
cell.a    = f"""{X}    0.0    0.0
                0.0    {Y}    0.0
                0.0    0.0    {2*R} """
cell.atom = f""" H {X/2}    {Y/2}    {R/2}
                H {X/2}    {Y/2}    {3*R/2} """
cell.basis = basis
cell.verbose = 5
cell.precision = 1e-12
cell.build(dump_input=False)


kmesh = [1, 1, 3]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts

#sets names for chkfile for r
fn_r = f"{fileroot}_r{r}.chk"
#searches for r-based chkfile
chk_r = os.path.isfile(os.path.join(current_directory, fn_r))
# Density fitting
exxdiv = None
gdf_fname = f"gdf_ints_{fileroot}_r{r}.h5" #add {os.getpid()} to run multiple in parallel
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.build()
gdf.force_dm_kbuild = True

# mean-field solver

kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-12


#logic for kmf.init_guess based on chkfile existence
if chk_r is True:               #looks for chkfile for same script and same k value
    chkfile = fn_r              
    kmf.init_guess = 'chkfile'  #if present, sets initial guess to that chkfile
else:                           #if no applicable chkfiles exist, sets the initial guess otherwise
    kmf.init_guess = 'vsap'               #'minao' as other option
    print("No chkfile available.")


#kmf = pbc_hp.smearing_(kmf, sigma=T)

dm0 = np.asarray(kmf.get_init_guess(cell, key="minao"))
dm0[0][0,0] = 2                                        #perturbation
    
kmf.kernel(dm0=dm0)
kmf.chkfile = f"{fileroot}_r{r}.chk"        #writes to a chkfile named after file and r value
#placed after kmf.kernel so only saved final cycle

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
        #solver = fted_solver.FTEDSolver(restricted=False, beta=beta, solve_mu=True)
        # solver = ftdmrg_solver.FTDMRGSolver(beta=beta, restricted=False, verbose=0)
        # solver = ltdmrg_solver.LTDMRGSolver(beta=beta, mu_gc=0, restricted=False, verbose=0)
        solver = fci_solver.FCISolver(restricted=False)
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
mydmet.init_guess = f"{fileroot}_dmet_r{r}.chk" 
mydmet.chkfile = f"{fileroot}_dmet_r{r}.chk" 
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
natm = cell.natm
Nalpha = np.zeros(natm)
Nbeta  = np.zeros(natm)

for k in range(nk):
    Sk = S[k]
    Da = dm[0, k]
    Db = dm[1, k]

    # Löwdin S^(1/2) for this k-point
    e, U = np.linalg.eigh(Sk)
    e = np.clip(e, 0.0, None)  # kill tiny negative eigenvalues from numerical noise
    S12 = (U * np.sqrt(e)) @ U.conj().T

    # Löwdin-orthogonal populations: diag(S^(1/2) D S^(1/2))
    Pa = S12 @ Da @ S12
    Pb = S12 @ Db @ S12
    pa = np.real(np.diag(Pa))
    pb = np.real(np.diag(Pb))

    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        Nalpha[A] += wk[k] * pa[p0:p1].sum()
        Nbeta[A]  += wk[k] * pb[p0:p1].sum()

m = Nalpha - Nbeta

M = float(m.sum())
abs_m = float(np.abs(m).sum())

eps = 1e-3
if abs_m < eps:
    label = "Paramagnetic"
elif abs(M) > 0.5 * abs_m:
    label = "Ferromagnetic"
elif abs(M) < 0.1 * abs_m and natm >= 2 and np.sign(m[0]) == -np.sign(m[1]):
    label = "AntiFerromagnetic"
else:
    label = "mixed/complex"

e_tot_mf = kmf.e_tot              # total MF energy (electronic + nuclear)
e_nuc    = cell.energy_nuc()      # nuclear energy from PySCF
e_elec_mf = e_tot_mf - e_nuc      # electronic-only part
print(f"{R_b} {e_elec_mf} {e_nuc} {e_tot_mf}")


e_nuc        = mydmet.h0              # same as KMF nuclear
e_tot_dmet   = mydmet.e_tot           # DMET total (elec + h0)
e_elec_dmet  = e_tot_dmet - e_nuc     # DMET electronic-only
print(f"{R_b} {e_elec_dmet/2} {e_nuc/2} {e_tot_dmet/2}")
# print("\n=== Per-atom Löwdin spin populations (k-averaged) ===")
# for A in range(natm):
#     sym = cell.atom_symbol(A)
#     print(f"Atom {A} ({sym}): Nα={Nalpha[A]: .12f}  Nβ={Nbeta[A]: .12f}  m={m[A]: .12f}")
# print(f"Total electrons (Σ(Nα+Nβ)) = {float(np.sum(Nalpha+Nbeta)):.12f}")
# print(f"Net magnetization M = Σ m_A = {M:.12f}")
# print(f"Spin order label = {label}")

# print(f"{R_b}   {m[0]} {m[1]}")
