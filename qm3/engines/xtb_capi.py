import  os
import  ctypes
import  typing
import  numpy
import  qm3.data
import  qm3.engines


class run( qm3.engines.template ):
    """
    GFN2-xTB/MM through the xtb C API (libxtb.so, xtb >= 6.3): no files, no external process,
    and the wavefunction of the previous call is kept as the guess of the next one

    chrg / nope         QM charge and number of unpaired electrons (uhf)
    opts:
        lib             path to libxtb.so (default: $CONDA_PREFIX/lib/libxtb.so)
        accuracy        xtb numerical accuracy (default 1.0, as the xtb binary)
        etemp           electronic temperature (K, default 300)
        max_iter        SCC iterations (default 250)
        pc_element      element whose GFN2 hardness smears the MM point charges (default 7, the
                        xtb pcem default "dummy atom")
        pc_cutoff       only MM charges within this distance (A) of any QM atom are embedded (None: all);
                        charges switch off smoothly over the last pc_switch A, so energy and forces
                        stay continuous when a charge crosses the cutoff
                        NOT VALIDATED: its gradient fails finite differences (errors ~0.5 kJ/(mol A) on QM
                        atoms and on MM charges in the switching region) and the truncation shifts the
                        QM/MM energy by tens to hundreds of kJ/mol; keep the default (all charges)
        pc_switch       width of the switching region (A, default 2.0)
        restart         reuse the previous wavefunction (default True)
        verbosity       xtb output level (0: silent, 1: SCC iterations)
        output          file for the xtb output (default: stdout)
    MM charges are taken in the minimum image around the QM centroid (img = True by default; the qm3 Fortran
    xtb engine wraps them around the origin instead).
    Load libxtb before importing pyscf: pyscf ships an older libdftd4.so with the same soname, which
    breaks libtblite ("undefined symbol __dftd4_cutoff_MOD_smooth_cutoff").
    Energy in kJ/mol, gradients in kJ/(mol A); link atoms through the exact chain rule (Link_chain).
    """
    def __init__( self, mol: object, chrg: int,
            nope: typing.Optional[int] = 0,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [],
            opts: typing.Optional[dict] = {} ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.opt = { "lib": os.path.join( os.environ.get( "CONDA_PREFIX", "" ), "lib", "libxtb.so" ),
                "accuracy": 1.0, "etemp": 300.0, "max_iter": 250, "pc_element": 7,
                "pc_cutoff": None, "pc_switch": 2.0, "restart": True, "verbosity": 0, "output": None }
        self.opt.update( opts )
        if( self.opt["pc_cutoff"] ):
            print( ">> qm3.engines.xtb_capi: pc_cutoff is NOT validated (gradient fails finite differences, energy shifted): use all charges" )
        self.cx  = 1.0 / qm3.data.A0                 # A -> Bohr
        self.ce  = qm3.data.H2J                      # Eh -> kJ/mol
        self.cg  = self.ce * self.cx                 # Eh/Bohr -> kJ/(mol A)
        self.chg = float( chrg )
        self.uhf = int( nope )
        self.nqm = len( self.sel ) + len( self.lnk )
        self.anu = numpy.array( list( mol.anum[self.sel] ) + [ 1 ] * len( self.lnk ), dtype = numpy.int32 )
        # redistribute MM-charge on the remaining atoms of the group
        self.__dq = numpy.zeros( mol.natm )
        for i,j in self.lnk:
            if( j in self.grp ):
                self.__dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
        # ----------------------------------------------------------
        L = ctypes.CDLL( self.opt["lib"] )
        P, D, I = ctypes.c_void_p, ctypes.POINTER( ctypes.c_double ), ctypes.POINTER( ctypes.c_int )
        sig = { "xtb_newEnvironment": ( P, [] ), "xtb_newCalculator": ( P, [] ), "xtb_newResults": ( P, [] ),
                "xtb_newMolecule": ( P, [ P, I, I, D, D, I, D, ctypes.POINTER( ctypes.c_bool ) ] ),
                "xtb_updateMolecule": ( None, [ P, P, D, D ] ),
                "xtb_loadGFN2xTB": ( None, [ P, P, P, ctypes.c_char_p ] ),
                "xtb_setExternalCharges": ( None, [ P, P, I, I, D, D ] ),
                "xtb_releaseExternalCharges": ( None, [ P, P ] ),
                "xtb_setAccuracy": ( None, [ P, P, ctypes.c_double ] ),
                "xtb_setMaxIter": ( None, [ P, P, ctypes.c_int ] ),
                "xtb_setElectronicTemp": ( None, [ P, P, ctypes.c_double ] ),
                "xtb_setVerbosity": ( None, [ P, ctypes.c_int ] ),
                "xtb_setOutput": ( None, [ P, ctypes.c_char_p ] ),
                "xtb_singlepoint": ( None, [ P, P, P, P ] ),
                "xtb_getEnergy": ( None, [ P, P, D ] ), "xtb_getGradient": ( None, [ P, P, D ] ),
                "xtb_getPCGradient": ( None, [ P, P, D ] ), "xtb_getCharges": ( None, [ P, P, D ] ),
                "xtb_checkEnvironment": ( ctypes.c_int, [ P ] ),
                "xtb_getError": ( None, [ P, ctypes.c_char_p, I ] ),
                "xtb_delResults": ( None, [ ctypes.POINTER( P ) ] ) }
        for k, ( r, a ) in sig.items():
            f = getattr( L, k ); f.restype = r; f.argtypes = a
        self.L = L
        self.env = L.xtb_newEnvironment()
        L.xtb_setVerbosity( self.env, self.opt["verbosity"] )
        if( self.opt["output"] ):
            L.xtb_setOutput( self.env, self.opt["output"].encode() )
        self.img = True
        self.mol = None
        self.calc = None
        self.res = None
        self.last_npc = 0


    def _ptr( self, a, t = ctypes.c_double ):
        return( a.ctypes.data_as( ctypes.POINTER( t ) ) )


    def _check( self, where ):
        if( self.L.xtb_checkEnvironment( self.env ) != 0 ):
            buf = ctypes.create_string_buffer( 2048 )
            self.L.xtb_getError( self.env, buf, ctypes.byref( ctypes.c_int( 2048 ) ) )
            raise RuntimeError( "qm3.engines.xtb_capi (%s): %s"%( where, buf.value.decode( errors = "ignore" ) ) )


    def _qm_coor( self, mol ):
        crd = [ mol.coor[i] for i in self.sel ]
        for i,j in self.lnk:
            c, v = qm3.engines.Link_coor( i, j, mol )
            crd.append( c )
        return( numpy.ascontiguousarray( numpy.array( crd ) * self.cx, dtype = numpy.float64 ) )


    def _mm( self, mol ):
        """MM charges (minimum image around the QM centroid) and, with pc_cutoff, their switching factors"""
        crd = mol.coor[self.nbn].copy()
        if( self.img ):
            cen = numpy.mean( mol.coor[self.sel], axis = 0 )
            crd -= mol.boxl * numpy.round( ( crd - cen ) / mol.boxl, 0 )
        chg = mol.chrg[self.nbn] + self.__dq[self.nbn]
        if( not self.opt["pc_cutoff"] ):
            return( numpy.arange( len( self.nbn ) ), crd, chg, None, None, None )
        rc, w = self.opt["pc_cutoff"], self.opt["pc_switch"]
        # distance to the closest real QM atom: sw( r ) = 1 - smoothstep( ( r - rc + w ) / w )
        qa  = mol.coor[self.sel]
        rr  = numpy.sqrt( ( ( crd[:,None,:] - qa[None,:,:] ) ** 2 ).sum( axis = 2 ) )
        jmn = numpy.argmin( rr, axis = 1 )
        r   = rr[numpy.arange( len( rr ) ), jmn]
        idx = numpy.flatnonzero( r < rc )
        r, jmn = r[idx], jmn[idx]
        x   = numpy.clip( ( r - ( rc - w ) ) / w, 0.0, 1.0 )
        sw  = 1.0 - x * x * x * ( 10.0 - 15.0 * x + 6.0 * x * x )
        dsw = - 30.0 * x * x * ( 1.0 - x ) ** 2 / w                               # d sw / d r (1/A)
        u   = ( crd[idx] - qa[jmn] ) / r[:,None]
        return( idx, crd[idx], chg[idx], sw, dsw[:,None] * u, jmn )


    def _setup( self, xyz ):
        n = ctypes.c_int( self.nqm ); chg = ctypes.c_double( self.chg ); uhf = ctypes.c_int( self.uhf )
        self.mol = self.L.xtb_newMolecule( self.env, ctypes.byref( n ), self._ptr( self.anu, ctypes.c_int ),
                self._ptr( xyz ), ctypes.byref( chg ), ctypes.byref( uhf ), None, None )
        self._check( "newMolecule" )
        self.calc = self.L.xtb_newCalculator()
        self.L.xtb_loadGFN2xTB( self.env, self.mol, self.calc, None )
        self._check( "loadGFN2xTB" )
        self.L.xtb_setAccuracy( self.env, self.calc, float( self.opt["accuracy"] ) )
        self.L.xtb_setMaxIter( self.env, self.calc, int( self.opt["max_iter"] ) )
        self.L.xtb_setElectronicTemp( self.env, self.calc, float( self.opt["etemp"] ) )
        self.res = self.L.xtb_newResults()


    def _calc( self, mol, grad ):
        xyz = self._qm_coor( mol )
        if( self.mol is None ):
            self._setup( xyz )
        else:
            self.L.xtb_updateMolecule( self.env, self.mol, self._ptr( xyz ), None )
            self._check( "updateMolecule" )
            if( not self.opt["restart"] ):
                ptr = ctypes.c_void_p( self.res )
                self.L.xtb_delResults( ctypes.byref( ptr ) )
                self.res = self.L.xtb_newResults()
        npc = 0
        if( len( self.nbn ) > 0 ):
            idx, crd, chg, sw, dsw, jmn = self._mm( mol )
            npc = len( idx )
            num = numpy.full( npc, int( self.opt["pc_element"] ), dtype = numpy.int32 )
            q   = numpy.ascontiguousarray( chg if sw is None else chg * sw, dtype = numpy.float64 )
            pos = numpy.ascontiguousarray( crd * self.cx, dtype = numpy.float64 )
            self.L.xtb_releaseExternalCharges( self.env, self.calc )
            self.L.xtb_setExternalCharges( self.env, self.calc, ctypes.byref( ctypes.c_int( npc ) ),
                    self._ptr( num, ctypes.c_int ), self._ptr( q ), self._ptr( pos ) )
            self._check( "setExternalCharges" )
        self.last_npc = npc
        self.L.xtb_singlepoint( self.env, self.mol, self.calc, self.res )
        self._check( "singlepoint" )
        e = ctypes.c_double()
        self.L.xtb_getEnergy( self.env, self.res, ctypes.byref( e ) )
        out = e.value * self.ce
        mol.func += out
        if( grad ):
            g = numpy.zeros( ( self.nqm, 3 ) )
            self.L.xtb_getGradient( self.env, self.res, self._ptr( g ) )
            g *= self.cg
            mol.grad[self.sel,:] += g[:len( self.sel )]
            qm3.engines.Link_chain( mol, self.lnk, g[len( self.sel ):] )
            if( npc > 0 ):
                gp = numpy.zeros( ( npc, 3 ) )
                self.L.xtb_getPCGradient( self.env, self.res, self._ptr( gp ) )
                mol.grad[self.nbn[idx],:] += gp * self.cg
                if( sw is not None ):
                    # E depends on q_i * sw( r_i ): dE/dr_i += q_i * phi_i * dsw/dr_i, with phi_i the QM
                    # potential at charge i (Hellmann-Feynman); in the switching region (r > rc - w) the
                    # QM atomic charges are far enough for phi = sum_A q_A / r_iA (no hardness damping)
                    f = ( chg * self._qm_potential( mol, crd ) )[:,None] * dsw
                    mol.grad[self.nbn[idx],:] += f
                    numpy.add.at( mol.grad, self.sel[jmn], - f )
        return( out )


    def _qm_potential( self, mol, crd ):
        """electrostatic potential (kJ/mol/e) of the QM atomic charges at crd (A)"""
        q = numpy.zeros( self.nqm )
        self.L.xtb_getCharges( self.env, self.res, self._ptr( q ) )
        qa = self._qm_coor( mol ) / self.cx
        r  = numpy.sqrt( ( ( crd[:,None,:] - qa[None,:,:] ) ** 2 ).sum( axis = 2 ) )
        return( numpy.sum( q[None,:] / r, axis = 1 ) * self.ce * qm3.data.A0 )


    def get_func( self, mol ):
        return( self._calc( mol, False ) )


    def get_grad( self, mol ):
        return( self._calc( mol, True ) )
