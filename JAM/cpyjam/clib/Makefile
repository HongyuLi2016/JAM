SHELL = /bin/sh

# ------------------------------------ #

# GSL libraries [path: -L/usr/local/lib is default]
GSLLIBS = -lgsl -lgslcblas

# combine all libraries
LIBS = -lm $(GSLLIBS)

# compiler: -g produces debugging information, -Wall turns on all warnings
CC = gcc -Wall -ffast-math -O3 -fomit-frame-pointer -std=gnu99
CFLAGS= -fPIC -I/share/apps/gsl/include/

INTERP = interp2dpol.o
INTERP := $(INTERP:%=interp/%)

JAM = jam_axi_rms_mgeint.o jam_axi_rms_mmt.o jam_axi_rms_wmmt.o \
	jam_axi_vel_losint.o jam_axi_vel_mgeint.o jam_axi_vel_mmt.o \
	jam_axi_vel_wmmt.o
JAM := $(JAM:%=jam/%)

MGE = mge_addbh.o mge_dens.o mge_deproject.o mge_qmed.o mge_surf.o
MGE := $(MGE:%=mge/%)

TOOLS = maximum.o median.o minimum.o range.o sort_dbl.o where.o
TOOLS := $(TOOLS:%=tools/%)


libcjam.so: $(INTERP) $(JAM) $(MGE) $(TOOLS) 
	$(CC) $(INTERP) $(JAM) $(MGE) $(TOOLS) -shared -o libcjam.so $(LIBS) -L/share/apps/gsl/lib 

clean: 
	rm  */*.o
