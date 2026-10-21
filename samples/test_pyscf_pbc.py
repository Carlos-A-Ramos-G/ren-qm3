import  os
os.environ["OMP_NUM_THREADS"] = "1"
import	numpy
import	qm3
import  qm3.engines.pyscf
import  sys
import  time

#
# periodic (Ewald) DFT/MM: analytic gradients vs central finite differences
#
#   python3 test_pyscf_pbc.py [cpu|gpu]
#

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep
gpu = len( sys.argv ) > 1 and sys.argv[1] == "gpu"


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
print( sqm.sum(), smm.sum() )

opt = { "basis": "sto-3g",
    "conv_tol": 1.e-11,
    "charge": 1,
    "spin": 0,
    "method": "lda,vwn",
    "memory": 4096, # MB
    "grid": 3,
    "max_cyc": 100,
    "nproc": 4,
    "rcut_hcore": 8.0 }
mol.engines["qm"] = qm3.engines.pyscf.run_pbc( mol, opt, sqm, smm, sla, gpu = gpu )

t0 = time.time()
mol.get_grad()
print( "gradient: %.1lf s"%( time.time() - t0 ) )
print( "E = %.4lf kJ/mol"%( mol.func ) )
ana = mol.grad.copy()

# QM atoms not involved in the link, MM atoms close to (exact hcore) and far from (multipoles/Ewald) the QM center
cen = numpy.mean( mol.coor[sqm], axis = 0 )
dsp = mol.coor - cen
dsp -= mol.boxl * numpy.round( dsp / mol.boxl, 0 )
dst = numpy.linalg.norm( dsp, axis = 1 )
lnk = [ sla[0][0], sla[0][1] ]
iqm = [ i for i in numpy.flatnonzero( sqm ) if i not in lnk ][:2]
imm = [ i for i in numpy.argsort( dst ) if smm[i] and i not in lnk ]
who = iqm + imm[:2] + [ i for i in imm if dst[i] > opt["rcut_hcore"] + 1.0 ][:2]

dsp = 1.e-4
err = 0.0
for i in who:
    for j in [0, 1, 2]:
        bak = mol.coor[i,j]
        mol.coor[i,j] = bak + dsp
        fp = mol.get_func()
        mol.coor[i,j] = bak - dsp
        fm = mol.get_func()
        mol.coor[i,j] = bak
        num = ( fp - fm ) / ( 2.0 * dsp )
        err = max( err, abs( num - ana[i,j] ) )
        print( "%5d %-4s %s %6.2lf %14.6lf %14.6lf %10.2e"%( i, mol.labl[i], "QM" if sqm[i] else "MM", dst[i],
            ana[i,j], num, num - ana[i,j] ) )
print( "max |analytic - numerical| = %.2e kJ/(mol A)"%( err ) )
