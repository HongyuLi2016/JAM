/* ----------------------------------------------------------------------------
  JAM_AXI_RMS_WMMT
    
    Calculates weighted second moment.
    
    INPUTS
      xp    : projected x' [pc]
      yp    : projected y' [pc]
      nxy   : number of x' and y' values given
      incl  : inclination [radians]
      lum3d   : 3d luminous MGE
      pot3d   : 3d potential MGE
      beta  : velocity anisotropy (1 - vz^2 / vr^2)
      vv    : velocity integral selector (1=xx, 2=yy, 3=zz, 4=xy, 5=xz, 6=yz)
    
    NOTES
    * Based on janis2_weighted_second_moment_squared IDL code by Michele
      Cappellari.
    
  Mark den Brok
  Laura L Watkins [lauralwatkins@gmail.com]
  
  This code is released under a BSD 2-clause license.
  If you use this code for your research, please cite:
  Watkins et al. 2013, MNRAS, 436, 2598
  "Discrete dynamical models of omega Centauri"
  http://adsabs.harvard.edu/abs/2013MNRAS.436.2598W
---------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include "jam.h"
#include "../mge/mge.h"
#include "../tools/tools.h"


// rjl - free off memory referred to in a struct multigaussexp object

static void kill_mge_memory(struct multigaussexp mge)
{
    free(mge.sigma);
    free(mge.area);
    free(mge.q);
}


double *jam_axi_rms_wmmt( double *xp, double *yp, int nxy, double incl, \
        struct multigaussexp *lum3d, struct multigaussexp *pot3d, \
        double *beta, int vv ) {
    // rjl performance improvement

    static int first_time = 1;
    static gsl_integration_workspace *w = NULL;

    // rjl - for smaller arrays it is cheaper to create them on stack rather than use malloc
    double kani[lum3d->ntotal], s2l[lum3d->ntotal], q2l[lum3d->ntotal], s2q2l[lum3d->ntotal];

    if (first_time)
    {
        first_time = 0;
        w = gsl_integration_workspace_alloc( 1000 );
	
    }


    
    // angles
    double ci = cos( incl );
    double si = sin( incl );
    
    // mge component combinations
    
    double s2p[pot3d->ntotal], e2p[pot3d->ntotal];
    
    // rjl performance imnprovement - using pow to square a number is slow !

    for (int i = 0; i < lum3d->ntotal; i++ ) {
        kani[i] = 1. / ( 1. - beta[i] );
        double is = lum3d->sigma[i];
        s2l[i] = is * is;
        double iq = lum3d->q[i];
        q2l[i] = iq * iq;
        s2q2l[i] = s2l[i] * q2l[i];
    }
    for (int i = 0; i < pot3d->ntotal; i++ ) {
        double is = pot3d->sigma[i];
        s2p[i] = is * is;
        double iq = pot3d->q[i];
        e2p[i] = 1. - iq * iq;
    }
    
    // parameters for the integrand function

    struct params_rmsint p;

    p.ci2 = ci * ci;
    p.si2 = si * si;
    p.cisi = ci * si;
    p.lum = lum3d;
    p.pot = pot3d;
    p.kani = kani;
    p.s2l = s2l;
    p.q2l = q2l;
    p.s2q2l = s2q2l;
    p.s2p = s2p;
    p.e2p = e2p;
    p.vv = vv;
    
    
    // perform integration
    
    gsl_function F;
    F.function = &jam_axi_rms_mgeint;
    F.params = &p;
    double *sb_mu2 = (double *) malloc( nxy * sizeof( double ) );

    for (int i = 0; i < nxy; i++ ) 
    {
        double result, error;

        p.x2 = xp[i] * xp[i];
        p.y2 = yp[i] * yp[i];
        p.xy = xp[i] * yp[i];
        gsl_integration_qag( &F, 0., 1., 1e-3, 1e-5, 1000, 2, w, &result, &error );

        sb_mu2[i] = result;
    }
    

    kill_mge_memory(*lum3d);
    kill_mge_memory(*pot3d);
    
    return sb_mu2;
    
}
