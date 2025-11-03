# Generated automatically from Makefile.in by configure.
#
#   This file is part of CONCORDE
#
#   (c) Copyright 1995--1999 by David Applegate, Robert Bixby,
#   Vasek Chvatal, and William Cook
#
#   Permission is granted for academic research use.  For other uses,
#   contact the authors for licensing options.
#
#   Use at your own risk.  We make no guarantees about the
#   correctness or usefulness of this code.
#


SHELL = /usr/bin/sh
SRCROOT = ..
BLDROOT = ..
CCINCDIR=$(SRCROOT)/INCLUDE

srcdir = .

CC = /home/rnarad/anaconda3/bin/x86_64-conda-linux-gnu-cc
CFLAGS = -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/rnarad/anaconda3/include  -I/home/rnarad/anaconda3/targets/x86_64-linux/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/rnarad/anaconda3/include  -I/home/rnarad/anaconda3/targets/x86_64-linux/include -I$(BLDROOT)/INCLUDE -I$(CCINCDIR)
LDFLAGS = -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/rnarad/anaconda3/include  -I/home/rnarad/anaconda3/targets/x86_64-linux/include -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,-rpath,/home/rnarad/anaconda3/lib -Wl,-rpath-link,/home/rnarad/anaconda3/lib -L/home/rnarad/anaconda3/lib  -L/home/rnarad/anaconda3/targets/x86_64-linux/lib -L/home/rnarad/anaconda3/targets/x86_64-linux/lib/stubs
LIBFLAGS = -liberty -lm 
RANLIB = /home/rnarad/anaconda3/bin/x86_64-conda-linux-gnu-ranlib

OBJ_SUFFIX = o
o = $(OBJ_SUFFIX)

EXE=pqtest
THISLIB=pq.a
LIBSRCS=pq.c     cuttree.c  consec1.c  necklace.c
ALLSRCS=pqtest.c $(LIBSRCS)

LIBS=$(BLDROOT)/UTIL/util.a

all: $(THISLIB)

everything: all pqtest

pqtest: pqtest.$o $(THISLIB) $(LIBS)
	$(CC) $(LDFLAGS) -o $@ pqtest.$o $(THISLIB) $(LIBS) $(LIBFLAGS)

clean:
	-rm -f *.$o $(THISLIB) pqtest

OBJS=$(LIBSRCS:.c=.o)

$(THISLIB): $(OBJS)
	$(AR) $(ARFLAGS) $(THISLIB) $(OBJS)
	$(RANLIB) $(THISLIB)

.PHONY: $(BLDROOT)/concorde.a
$(BLDROOT)/concorde.a: $(OBJS)
	$(AR) $(ARFLAGS) $(BLDROOT)/concorde.a $(OBJS)
	$(RANLIB) $(BLDROOT)/concorde.a

include ../INCLUDE/Makefile.common

# DO NOT DELETE THIS LINE -- make depend depends on it.

I=$(CCINCDIR)
I2=$(BLDROOT)/INCLUDE

consec1.$o:  consec1.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/tsp.h      $(I)/edgegen.h  $(I)/bigguy.h   $(I)/lp.h       \
        $(I)/cut.h      $(I)/kdtree.h   $(I)/combs.h    $(I)/pq.h       \
        $(I)/pqsets.h   $(I)/cuttree.h  $(I)/consec1.h  
cuttree.$o:  cuttree.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/tsp.h      $(I)/edgegen.h  $(I)/bigguy.h   $(I)/lp.h       \
        $(I)/cut.h      $(I)/kdtree.h   $(I)/combs.h    $(I)/pq.h       \
        $(I)/pqsets.h   $(I)/cuttree.h  
necklace.$o: necklace.c $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/tsp.h      $(I)/edgegen.h  $(I)/bigguy.h   $(I)/lp.h       \
        $(I)/cut.h      $(I)/kdtree.h   $(I)/combs.h    $(I)/pq.h       \
        $(I)/pqsets.h   $(I)/cuttree.h  $(I)/necklace.h $(I)/verify.h   \
        $(I)/macrorus.h 
pq.$o:       pq.c       $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/pq.h       $(I)/pqsets.h   $(I)/tsp.h      $(I)/edgegen.h  \
        $(I)/bigguy.h   $(I)/lp.h       $(I)/cut.h      $(I)/kdtree.h   \
        $(I)/combs.h    $(I)/cuttree.h  $(I)/macrorus.h 
pqtest.$o:   pqtest.c   $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/pq.h       $(I)/pqsets.h   
