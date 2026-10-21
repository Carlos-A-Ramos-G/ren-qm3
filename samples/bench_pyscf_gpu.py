import  os
os.environ["OMP_NUM_THREADS"] = "8"
import	numpy
import	qm3
import  qm3.engines.pyscf
import  sys
import  time

#
# GPU4PySCF periodic DFT/MM engine:
#   1. GPU vs CPU energy and gradients (same geometry)
#   2. time per gradient and SCF cycles along a short trajectory, reusing (or not) the previous density
#
#   python3 bench_pyscf_gpu.py [n_steps] [skip_cpu]
#

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep
nstp = int( sys.argv[1] ) if len( sys.argv ) > 1 else 10
skip = len( sys.argv ) > 2


def setup( gpu ):
    mol = qm3.molecule()
    mol.pdb_read( open( cwd + "charmm.pdb" ) )
    mol.psf_read( open( cwd + "charmm.psf" ) )
    mol.guess_atomic_numbers()
    mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )
    sqm = mol.resn == "WAT"
    for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
        sqm[mol.indx["A"][1][a]] = True
    sqm = numpy.logical_not( sqm )
    smm = numpy.logical_not( sqm )
    sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]
    opt = { "basis": "def2-svp", "conv_tol": 1.e-9, "charge": 1, "spin": 0, "method": "b3lypg",
        "memory": 16000, "grid": 3, "max_cyc": 100, "nproc": 8, "rcut_hcore": 10.0 }
    mol.engines["qm"] = qm3.engines.pyscf.run_pbc( mol, opt, sqm, smm, sla, gpu = gpu )
    return( mol, sqm )


def cycles( mol ):
    return( mol.engines["qm"].last_cycles )


# ------------------------------------------------------------------------------------
# 1. agreement
gmol, sqm = setup( True )
t0 = time.time()
gmol.get_grad()
print( ">> GPU first gradient: %.1lf s (includes warm-up)"%( time.time() - t0 ) )
if( not skip ):
    cmol, _ = setup( False )
    t0 = time.time()
    cmol.get_grad()
    print( ">> CPU gradient: %.1lf s"%( time.time() - t0 ) )
    print( ">> |E_gpu - E_cpu| = %.2e kJ/mol"%( abs( gmol.func - cmol.func ) ) )
    dif = numpy.abs( gmol.grad - cmol.grad )
    print( ">> max |G_gpu - G_cpu| QM = %.2e  MM = %.2e kJ/(mol A)"%( dif[sqm].max(), dif[~sqm].max() ) )

# ------------------------------------------------------------------------------------
# 2. trajectory-like displacements (~0.01 A per step)
rng = numpy.random.default_rng( 1234 )
ref = gmol.coor.copy()
for reuse in [ True, False ]:
    gmol.coor = ref.copy()
    gmol.engines["qm"].dm = None
    gmol.get_grad()
    tim = []
    for i in range( nstp ):
        gmol.coor += rng.normal( 0.0, 0.01, gmol.coor.shape )
        if( not reuse ):
            gmol.engines["qm"].dm = None
        t0 = time.time()
        gmol.get_grad()
        tim.append( time.time() - t0 )
        print( "   step %3d  %8.2lf s  scf_cycles: %s"%( i, tim[-1], cycles( gmol ) ) )
    print( ">> reuse density = %s: %.2lf s / gradient"%( reuse, numpy.mean( tim ) ) )
