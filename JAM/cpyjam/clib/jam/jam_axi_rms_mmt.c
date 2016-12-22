/* ----------------------------------------------------------------------------
  JAM_AXI_RMS_MMT
    
    Calculates second moment.
    
    INPUTS
      xp    : projected x' [pc]
      yp    : projected y' [pc]
      nxy   : number of x' and y' values given
      incl  : inclination [radians]
      lum   : projected luminous MGE
      pot   : projected potential MGE
      beta  : velocity anisotropy (1 - vz^2 / vr^2)
      nrad  : number of radial bins in interpolation grid
      nang  : number of angular bins in interpolation grid
      vv    : velocity integral selector (1=xx, 2=yy, 3=zz, 4=xy, 5=xz, 6=yz)
    
    NOTES
      * Based on janis2_second_moment IDL code by Michele Cappellari.
      * This version does not implement PDF convolution.
    
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
#include "jam.h"
#include "../mge/mge.h"
#include "../tools/tools.h"
#include "../interp/interp.h"


double* jam_axi_rms_mmt( double *xp, double *yp, int nxy, double incl, \
        struct multigaussexp *lum, struct multigaussexp *pot, double *beta, \
        double pixsize, double sigmapsf, int nrad, int nang, int vv ) {
    //printf("welcome!\n"); 
    int i, j, k;
    static int first_time = 1, npol;
    static int psfconvolve;
    static double  *surf, *mu, *xpol, *ypol, **mupol,*wm2;
    static double  qmed,*rell,step,rmax, *lograd, *rad, *ang, *angvec, *r, *e;
    if (first_time){ 
      // only set for the first time
      psfconvolve=(pixsize!=0. && sigmapsf!=0);
    }
    
    // skip the interpolation when computing just a few points
    if ( nrad * nang > nxy && !psfconvolve) {
        //printf("no interpolation!\n");
        // weighted second moment
        wm2 = jam_axi_rms_wmmt( xp, yp, nxy, incl, lum, pot, beta, vv );
        
        if ( vv == 4 ) {
            for ( i = 0; i < nxy; i++ ) {
                if ( xp[i] * yp[i] >= 0. ) wm2[i] *= -1.;
            }
        }
        
        if ( vv == 5 ) {
            for ( i = 0; i < nxy; i++ ) {
                if ( xp[i] * yp[i] < 0. ) wm2[i] *= -1.;
            }
        }
        
        // surface brightness
        if (first_time){
          //lhy only set for the first time
          surf = mge_surf( lum, xp, yp, nxy );
          mu = (double *) malloc( nxy * sizeof( double ) );
          first_time=0;
        }
        
        // second moment        
        for ( i = 0; i < nxy; i++ ) {
            mu[i] = wm2[i] / surf[i];
        }
        
        //free( wm2 ); //lhy already static, see jam_axi_rms_wmmt.c
        
        return mu;
    }
    else{
      if (first_time){    
        //lhy only create grid for the first time
        // elliptical radius of input (x,y)
        qmed = mge_qmed( lum, maximum( xp, nxy ) );
        rell = (double *) malloc( nxy * sizeof( double ) );
        for ( i = 0; i < nxy; i++ ) \
            rell[i] = sqrt( xp[i] * xp[i] + yp[i] * yp[i] / qmed / qmed );
         
        // set interpolation grid parameters
        step = minimum( rell, nxy );
        if ( step <= 10. ) step = 10.;          // minimum radius of 10. pc
        npol = nrad * nang;
    
        // make linear grid in log of elliptical radius
        rmax = maximum( rell, nxy );
        lograd = range( log( step ) - 0.1, log( rmax ) + 0.1, nrad, False ); 
        rad = (double *) malloc( nrad * sizeof( double ) );
        for ( i = 0; i < nrad; i++ ) rad[i] = exp( lograd[i] );
    
        // make linear grid in eccentric anomaly
        ang = range( -M_PI, -M_PI / 2., nang, False );
        angvec = range( -M_PI, M_PI, 4 * nang - 3, False );
    
        // convert grid to cartesians
        xpol = (double *) malloc( npol * sizeof( double ) );
        ypol = (double *) malloc( npol * sizeof( double ) );
        for ( i = 0; i < nrad; i++ ) {
            for ( j = 0; j < nang; j++ ) {
                xpol[i*nang+j] = rad[i] * cos( ang[j] );
                ypol[i*nang+j] = rad[i] * sin( ang[j] ) * qmed;
            }
        }
    
        // set up interpolation grid arrays
        mupol = (double **) malloc( nrad * sizeof( double * ) );
        for ( i = 0; i < nrad; i++ ) \
            mupol[i] = (double *) malloc( ( 4 * nang - 3 ) * sizeof( double ) );
        // first_time=0 is below, do not worry
      }
    
    // ---------------------------------
    
    
      // weighted second moment on polar grid
      wm2 = jam_axi_rms_wmmt( xpol, ypol, npol, incl, lum, pot, beta, vv );
    
      if (first_time){
        //lhy only calculate for the first time
        // surface brightness on polar grid
        surf = mge_surf( lum, xpol, ypol, npol );
        // elliptical radius and eccentric anomaly of inputs
        r = (double *) malloc( nxy * sizeof( double ) );
        e = (double *) malloc( nxy * sizeof( double ) );
        for ( i = 0; i < nxy; i++ ) {
            r[i] = sqrt( pow( xp[i], 2. ) + pow( yp[i] / qmed, 2. ) );
            e[i] = atan2( yp[i] / qmed, xp[i] );
        }
        first_time=0;
      }

      if (!psfconvolve){
        //printf("no psf!\n");
        // model velocity on the polar grid
        for ( i = 0; i < nrad; i++ ) {
            for ( j = 0; j < nang; j++ ) {
                k = i * nang + j;
                mupol[i][j] = wm2[k] / surf[k];
                mupol[i][2*nang-2-j] = mupol[i][j];
                mupol[i][2*nang-2+j] = mupol[i][j];
                mupol[i][4*nang-4-j] = mupol[i][j];
            }
        }
    
        mu = interp2dpol( mupol, rad, angvec, r, e, nrad, 4*nang-3, nxy );
    
        // fix signs of xy and xz second moments
        if ( vv == 4 ) for ( i = 0; i < nxy; i++ ) \
            if ( xp[i] * yp[i] >= 0. ) mu[i] *= -1.;
    
        if ( vv == 5 ) for ( i = 0; i < nxy; i++ ) \
            if ( xp[i] * yp[i] < 0. ) mu[i] *= -1.;
        return mu;
      }
      else{
        //printf("psf!\n");
        for ( i = 0; i < nrad; i++ ) {
            for ( j = 0; j < nang; j++ ) {
                k = i * nang + j;
                mupol[i][j] = wm2[k]; // / surf[k]; 
                //return surface brightness weighted value for PSF convolve
                mupol[i][2*nang-2-j] = mupol[i][j];
                mupol[i][2*nang-2+j] = mupol[i][j];
                mupol[i][4*nang-4-j] = mupol[i][j];
            }
        }
        mu = interp2dpol( mupol, rad, angvec, r, e, nrad, 4*nang-3, nxy );
        //printf("lhy\n");
        //mu=r;
        // fix signs of xy and xz second moments
        if ( vv == 4 ) for ( i = 0; i < nxy; i++ ) \
            if ( xp[i] * yp[i] >= 0. ) mu[i] *= -1.;
    
        if ( vv == 5 ) for ( i = 0; i < nxy; i++ ) \
            if ( xp[i] * yp[i] < 0. ) mu[i] *= -1.;
        return mu;
      }
    //lhy this is the memory need to be free in the original version
         //see the explanation below why they do not need to be free now    
    //free( rell ); //only allocate once here, set static
    //free( rad );  //only create once here (by function range), set static
    //free( ang );  //only create once here (by function range), set static
    //free( angvec ); //only create once here (by function range), set static
    //free( xpol );  //only allocate once here, set static
    //free( ypol );  //only allocate once here, set static
    //for ( i = 0; i < nrad; i++ ) free( mupol[i] ); //only allocate once here, set static
    //free( mupol ); //only allocate once here, set static
    //free( wm2 );   // set static in jam_axi_rms_wmmt.c, I think this is the leak before
    //free( surf );  //only create once here (by function mge_surf), set static
    //free( r ); //only allocate once here, set static
    //free( e ); //only allocate once here, set static
    // lhy mu should be freed in cython level, but I did not do that before, this 
    // is also the reason of leak, solve the problem by setting it as static in interp2dpol
      
    }
}
