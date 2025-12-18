import os
import numpy as np
import csv
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
basis = 'sto6g'        #'321g'  'sto6g'  '631g'
#R_b = float(os.environ.get("R_b", "3.4"))
R_b = 1.5
R=R_b * 0.529177249
r = round(R,1)
cell = gto.Cell()

pairs = 2               #number of hydrogen pairs
nH = 2 * pairs              
X = 30
Y = X
Z = nH * R          #
x = 0
y = x
cell.unit = 'Angstrom' #Is already this by default, added for clarity
cell.a    = f"""{X}    0.0    0.0
                0.0    {Y}    0.0
                0.0    0.0    {Z} """
cell.atom = "\n".join(
    f"H {x}    {y}    {(i + 0.5)*R}"
    for i in range(nH)
)
cell.basis = basis
cell.verbose = 5
cell.precision = 1e-12
cell.build(dump_input=False)

k = 3
kmesh = [1, 1, k]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts

#sets names for chkfile for r
fn_r = f"{fileroot}_r{r}.chk"
fn_dr = f"{fileroot}_dmet_r{r}.chk"
#searches for r-based chkfile
chk_r = os.path.isfile(os.path.join(current_directory, fn_r))
chk_dr = os.path.isfile(os.path.join(current_directory, fn_dr))
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
kmf.max_cycle = 400

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
beta = 1/T
solver = cc_solver.CCSolver(restricted=False, restart=True, verbose=cell.verbose, beta=beta)
slv = 'cc'
solver_argss = [["C_lo_eo", ("fname_cc", "fucc.h5")]]
# if T < 0.01:
        # beta = 1000
# else:
        # beta = 1./T
# if beta > 100:
        # solver = cc_solver.CCSolver(restricted=False, restart=True, verbose=cell.verbose, beta=beta)
        #slv = 'cc'
        # solver_argss = [["C_lo_eo", ("fname_cc", "fucc.h5")]]
# # 
# # solver = fci_solver.FCISolver(restricted=False)
    ##slv = 'fci'
# else:
        # #solver = fted_solver.FTEDSolver(restricted=False, beta=beta, solve_mu=True)
        #slv = 'fted'
        # # solver = ftdmrg_solver.FTDMRGSolver(beta=beta, restricted=False, verbose=0)
            #slv = 'ftdmrg'
        # # solver = ltdmrg_solver.LTDMRGSolver(beta=beta, mu_gc=0, restricted=False, verbose=0)
            #slv = 'ltdmrg'
        # #solver = fci_solver.FCISolver(restricted=False)
        #slv = 'fci'
        # solver_argss = None
        
# solve dmet

    

mydmet = udmet.UDMET(Lat, kmf, solver, C_ao_lo, vcor=None,
                        solver_argss=solver_argss)
mydmet.dump_flags()

# get the global Mu for finite temperature 
res = mydmet.solve_mf(return_mu=True)
ft_mu = res[-1]

#mydmet.calc_e_tot = False
cycles = 20
mydmet.max_cycle = cycles
mydmet.mu_glob = np.array(ft_mu, dtype=float)
mydmet.beta = beta
mydmet.bath_power = 1
mydmet.fit_method = 'CG'
mydmet.fit_kwargs = {"test_grad": True}
if chk_r is True:               #looks for chkfile for same script and same k value
    chkfile = fn_dr              
    mydmet.init_guess = 'chkfile'  #if present, sets initial guess to that chkfile
else:                           #if no applicable chkfiles exist, sets the initial guess otherwise
    mydmet.init_guess = 'vsap'               #'minao' as other option
    print("No chkfile available.")
mydmet.chkfile = fn_dr
mydmet.kernel()
#build density matrix from dmet calculations

Vcorr = mydmet.vcor
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
print(f"{R_b} {e_elec_mf/nH} {e_nuc/nH} {e_tot_mf/nH}")

runname = f"{R_b}_{nH}_{k+1}_{cycles}_{basis}_{slv}"  #makes name for csv entry with specs







e_nuc        = mydmet.h0              # same as KMF nuclear
e_tot_dmet   = mydmet.e_tot           # DMET total (elec + h0)
e_elec_dmet  = e_tot_dmet - e_nuc     # DMET electronic-only
print(f"{R_b} {e_elec_dmet/nH} {e_nuc/nH} {e_tot_dmet/nH}")

e_tot_perH = e_tot_dmet / nH
csv_path = os.path.join(current_directory, "results.csv")
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["Run Name", "e_tot_perH"])
    w.writerow([runname, float(e_tot_perH)])
print("\n=== Per-atom Löwdin spin populations (k-averaged) ===")
for A in range(natm):
     sym = cell.atom_symbol(A)
     print(f"Atom {A} ({sym}): Nα={Nalpha[A]: .12f}  Nβ={Nbeta[A]: .12f}  m={m[A]: .12f}")
print(f"Total electrons (Σ(Nα+Nβ)) = {float(np.sum(Nalpha+Nbeta)):.12f}")
print(f"Net magnetization M = Σ m_A = {M:.12f}")
print(f"Spin order label = {label}")

print(f"{R_b}   {m[0]} {m[1]}")

print(f"{Vcorr}")

vcor_path = os.path.join(current_directory, f"hchain_Vcor_{runname}.txt")
with open(vcor_path, "w") as f:
    print(Vcorr, file=f)
print(f"Wrote Vcor to: {vcor_path}")
#dumps Vcor into a txt file for future reference

rdm1_path = os.path.join(current_directory, f"hchain_rdm1_{runname}.txt")
with open(rdm1_path, "w") as f:
    print(dm, file=f)
print(f"Wrote rdm1 to: {rdm1_path}")
#dumps RDM1 to txt file for future reference