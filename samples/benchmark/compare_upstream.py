#!/usr/bin/env python3
"""
Benchmark and cross-check: upstream runGPU (pyscf_upstream.py, verbatim) vs the new run_pbc engine

engines (same system, same geometries):
    upstream          original runGPU as published
    upstream+unitfix  original runGPU with only the MM-coordinate unit bug removed
    new-matched       run_pbc with the physics of the original (no density fitting, no grid response)
    new               run_pbc with its defaults (density fitting + grid response)

checks:
    1. energy / gradient differences between engines at the reference geometry
    2. analytic gradient vs central finite differences (per engine) on QM atoms and on
       MM atoms inside and beyond rcut_hcore
    3. time per gradient (and SCF cycles for run_pbc) along a short jittered trajectory,
       plus the energy differences along it

usage (from the repository root):
    PYTHONPATH=$PWD python3 samples/benchmark/compare_upstream.py --backend gpu
    PYTHONPATH=$PWD python3 samples/benchmark/compare_upstream.py --backend cpu --level quick   # local check

--backend cpu runs a mechanical CPU port of the upstream code (gpu4pyscf -> pyscf); it is only a
stand-in to validate the harness where no GPU is available: the real comparison is --backend gpu
"""
import  os
import  sys
import  argparse
import  json
import  re
import  shutil
import  tempfile
import  time
import  types

ap = argparse.ArgumentParser()
ap.add_argument( "--backend", choices = [ "gpu", "cpu" ], default = "gpu" )
ap.add_argument( "--level", choices = [ "production", "quick" ], default = "production",
        help = "production: B3LYP/def2-SVP; quick: LDA/STO-3G" )
ap.add_argument( "--steps", type = int, default = 10, help = "trajectory steps for timing" )
ap.add_argument( "--jitter", type = float, default = 0.01, help = "random displacement per step (Ang)" )
ap.add_argument( "--fd", type = int, default = 1, help = "atoms per class for finite differences (0: skip)" )
ap.add_argument( "--engines", default = "upstream,upstream+unitfix,new-matched,new" )
ap.add_argument( "--nproc", type = int, default = 8 )
ap.add_argument( "--json", default = "compare_upstream.json" )
arg = ap.parse_args()
arg.json = os.path.abspath( arg.json )

os.environ.setdefault( "OMP_NUM_THREADS", str( arg.nproc ) )
import  numpy
import  qm3
import  qm3.engines
import  qm3.engines.pyscf

here = os.path.abspath( os.path.dirname( __file__ ) ) + os.sep
smpl = os.path.dirname( os.path.dirname( here ) ) + os.sep


# ------------------------------------------------------------------------------------
def load_upstream( unitfix: bool ):
    src = open( here + "pyscf_upstream.py" ).read()
    if( unitfix ):
        # runGPU: keep the MM coordinates in Angstrom (add_mm_charges is called without unit=, i.e. Angstrom)
        src, n = re.subn( r"\n( +)crd \*= self\.cx\n(\1chg = mol\.chrg\[self\.nbn\] \+ self\.__dq\[self\.nbn\]\n\1cut = )",
                r"\n\2", src )
        assert n == 1, "unit fix: pattern not found"
    if( arg.backend == "cpu" ):
        for a, b in [ ( "import  gpu4pyscf.dft\n", "import  pyscf.dft\n" ),
                ( "import  gpu4pyscf.qmmm.pbc\n", "import  pyscf.qmmm.pbc.itrf\n" ),
                ( "gpu4pyscf.dft.", "pyscf.dft." ),
                ( "gpu4pyscf.qmmm.pbc.itrf.", "pyscf.qmmm.pbc.itrf." ),
                ( "scf.make_rdm1().get()", "numpy.asarray( scf.make_rdm1() )" ) ]:
            assert a in src, "cpu port: '%s' not found"%( a.strip() )
            src = src.replace( a, b )
    else:
        import  gpu4pyscf.dft
        import  gpu4pyscf.qmmm.pbc
    mod = types.ModuleType( "pyscf_upstream" )
    exec( compile( src, here + "pyscf_upstream.py", "exec" ), mod.__dict__ )
    assert hasattr( mod, "runGPU" ), "upstream runGPU not defined (import error swallowed by its try/except)"
    return( mod.runGPU )


def system():
    mol = qm3.molecule()
    mol.pdb_read( open( smpl + "charmm.pdb" ) )
    mol.psf_read( open( smpl + "charmm.psf" ) )
    mol.guess_atomic_numbers()
    mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )
    sqm = mol.resn == "WAT"
    for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
        sqm[mol.indx["A"][1][a]] = True
    sqm = numpy.logical_not( sqm )
    sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]
    return( mol, sqm, numpy.logical_not( sqm ), sla )


if( arg.level == "production" ):
    base = { "basis": "def2-svp", "method": "b3lypg" }
else:
    base = { "basis": "sto-3g", "method": "lda,vwn" }
base.update( { "conv_tol": 1.e-10, "charge": 1, "spin": 0, "memory": 16000, "grid": 3, "max_cyc": 100, "nproc": arg.nproc } )


def engine( name: str, mol, sqm, smm, sla ):
    if( name == "upstream" ):
        return( load_upstream( False )( mol, dict( base ), sqm, smm, sla ) )
    if( name == "upstream+unitfix" ):
        return( load_upstream( True )( mol, dict( base ), sqm, smm, sla ) )
    gpu = arg.backend == "gpu"
    if( name == "new-matched" ):
        opt = dict( base, density_fit = False, grid_response = False )
        return( qm3.engines.pyscf.run_pbc( mol, opt, sqm, smm, sla, gpu = gpu ) )
    if( name == "new" ):
        return( qm3.engines.pyscf.run_pbc( mol, dict( base ), sqm, smm, sla, gpu = gpu ) )
    raise ValueError( name )


# ------------------------------------------------------------------------------------
names = arg.engines.split( "," )
mol, sqm, smm, sla = system()
ref = mol.coor.copy()
lnk = [ sla[0][0], sla[0][1] ]

# finite-difference targets: QM atoms off the link, MM atoms near / beyond rcut_hcore (half box: the upstream value)
rhc = 0.5 * numpy.min( mol.boxl )
cen = numpy.mean( ref[sqm], axis = 0 )
dsp = ref - cen
dsp -= mol.boxl * numpy.round( dsp / mol.boxl, 0 )
dst = numpy.linalg.norm( dsp, axis = 1 )
imm = [ i for i in numpy.argsort( dst ) if smm[i] and i not in lnk ]
fd_atoms = [ i for i in numpy.flatnonzero( sqm ) if i not in lnk ][:arg.fd] + imm[:arg.fd] + [ i for i in imm if dst[i] > rhc ][:arg.fd]

res = {}
wrk = tempfile.mkdtemp( prefix = "qm3_cmp_" )
for name in names:
    print( "\n==================== %s (%s)"%( name, arg.backend ) )
    # own folder: the upstream engine keeps its guess in ./pyscf.chk
    os.chdir( wrk ); os.makedirs( name, exist_ok = True ); os.chdir( name )
    mol.coor = ref.copy()
    mol.engines = { "qm": engine( name, mol, sqm, smm, sla ) }
    out = { "fd": [] }

    t0 = time.time()
    mol.get_grad()
    out["t_first"] = time.time() - t0
    out["E"] = mol.func
    out["G"] = mol.grad.copy()
    print( "E = %.6lf kJ/mol   first gradient %.2lf s"%( mol.func, out["t_first"] ) )

    for i in fd_atoms:
        for j in [0, 1, 2]:
            bak = mol.coor[i,j]
            mol.coor[i,j] = bak + 1.e-4
            fp = mol.get_func()
            mol.coor[i,j] = bak - 1.e-4
            fm = mol.get_func()
            mol.coor[i,j] = bak
            num = ( fp - fm ) / 2.e-4
            kind = "QM" if sqm[i] else ( "MM<rc" if dst[i] <= rhc else "MM>rc" )
            out["fd"].append( ( int( i ), j, kind, out["G"][i,j], num ) )
            print( "  fd %5d %-5s %d %14.6lf %14.6lf %10.2e"%( i, kind, j, out["G"][i,j], num, out["G"][i,j] - num ) )

    rng = numpy.random.default_rng( 1234 )
    mol.coor = ref.copy()
    mol.get_grad()
    out["t_step"] = []
    out["E_traj"] = []
    out["cycles"] = []
    for k in range( arg.steps ):
        mol.coor += rng.normal( 0.0, arg.jitter, mol.coor.shape )
        t0 = time.time()
        mol.get_grad()
        out["t_step"].append( time.time() - t0 )
        out["E_traj"].append( mol.func )
        out["cycles"].append( getattr( mol.engines["qm"], "last_cycles", None ) )
        print( "  step %3d %10.2lf s  E = %.6lf  cycles: %s"%( k, out["t_step"][-1], mol.func, out["cycles"][-1] ) )
    res[name] = out
os.chdir( here )
shutil.rmtree( wrk, ignore_errors = True )


# ------------------------------------------------------------------------------------
def fd_err( out, kind ):
    e = [ abs( a - n ) for i, j, k, a, n in out["fd"] if k == kind ]
    return( "%9.1e"%( max( e ) ) if e else "      n/a" )

print( "\n==================== summary (%s, %s, %d QM / %d MM atoms)"%( arg.backend, arg.level, sqm.sum(), smm.sum() ) )
print( "%-17s %10s %10s %7s | max |analytic - fd| kJ/(mol A)"%( "engine", "s/grad", "speedup", "cycles" ) )
print( "%-17s %10s %10s %7s | %9s %9s %9s"%( "", "", "", "", "QM", "MM<rc", "MM>rc" ) )
t_ref = numpy.mean( res[names[0]]["t_step"] ) if arg.steps > 0 else None
for name in names:
    out = res[name]
    t = numpy.mean( out["t_step"] ) if arg.steps > 0 else float( "nan" )
    cyc = [ c for c in out["cycles"] if c is not None ]
    print( "%-17s %10.2lf %9.2lfx %7s | %s %s %s"%( name, t, t_ref / t if t_ref else float( "nan" ),
        "%.1lf"%( numpy.mean( cyc ) ) if cyc else "-",
        fd_err( out, "QM" ), fd_err( out, "MM<rc" ), fd_err( out, "MM>rc" ) ) )

print( "\n%-36s %12s %12s %12s %12s"%( "pair", "|dE| ref", "max|dE| traj", "max|dG| QM", "max|dG| MM" ) )
pairs = [ ( "upstream+unitfix", "new-matched" ), ( "upstream", "upstream+unitfix" ),
        ( "new-matched", "new" ), ( "upstream", "new" ) ]
for a, b in pairs:
    if( a in res and b in res ):
        dg = numpy.abs( res[a]["G"] - res[b]["G"] )
        dt = numpy.abs( numpy.array( res[a]["E_traj"] ) - numpy.array( res[b]["E_traj"] ) )
        print( "%-36s %12.2e %12.2e %12.2e %12.2e"%( a + " vs " + b, abs( res[a]["E"] - res[b]["E"] ),
            dt.max() if dt.size else float( "nan" ), dg[sqm].max(), dg[smm].max() ) )
print( "(energies in kJ/mol, gradients in kJ/(mol A))" )

with open( arg.json, "wt" ) as f:
    json.dump( { "args": vars( arg ), "results": { n: { k: ( v.tolist() if isinstance( v, numpy.ndarray ) else v )
        for k, v in o.items() } for n, o in res.items() } }, f, indent = 1 )
print( "\n>> details saved to %s"%( os.path.abspath( arg.json ) ) )
