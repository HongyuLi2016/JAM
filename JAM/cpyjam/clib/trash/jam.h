/* -----------------------------------------------------------------------------
  JAM PROGRAMS
    
    jam_axi_rms_mgeint  : integrand for second moments
    params_rmsint       : parameter structure for second moment intergration
    multigaussexp : MGE structure
----------------------------------------------------------------------------- */


// definitions

#define True 1
#define False 0

#define AU  1.4959787068E8          // AU in km
#define G 0.00430237                // G in (km/s)^2 pc/Msun
#define RADEG 57.29578              // degrees per radian
#define RA2DEG 57.29578             // degrees per radian
#define pc2km  3.0856776e+13        // (km per parsec)


// ----------------------------------------------------------------------------


// structs

struct params_rmsint {
    struct multigaussexp *lum, *pot;
    double *kani, *s2l, *q2l, *s2q2l, *s2p, *e2p;
    double x2, y2, xy, ci, si, ci2, si2, cisi;
    int vv;
};

struct multigaussexp {
    double *area;
    double *sigma;
    double *q;
    int ntotal;
};


// ----------------------------------------------------------------------------


// programs

double jam_axi_rms_mgeint( double, void * );
