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
SRCROOT = .
BLDROOT = .
CCINCDIR=$(SRCROOT)/INCLUDE
BLDINCDIR=$(BLDROOT)/INCLUDE

srcdir = .

CC = /home/rnarad/anaconda3/bin/x86_64-conda-linux-gnu-cc
CFLAGS = -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/rnarad/anaconda3/include  -I/home/rnarad/anaconda3/targets/x86_64-linux/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/rnarad/anaconda3/include  -I/home/rnarad/anaconda3/targets/x86_64-linux/include -I$(BLDROOT)/INCLUDE -I$(CCINCDIR)
LDFLAGS = -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/rnarad/anaconda3/include  -I/home/rnarad/anaconda3/targets/x86_64-linux/include -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,-rpath,/home/rnarad/anaconda3/lib -Wl,-rpath-link,/home/rnarad/anaconda3/lib -L/home/rnarad/anaconda3/lib  -L/home/rnarad/anaconda3/targets/x86_64-linux/lib -L/home/rnarad/anaconda3/targets/x86_64-linux/lib/stubs
LIBFLAGS = -liberty -lm 
RANLIB = /home/rnarad/anaconda3/bin/x86_64-conda-linux-gnu-ranlib

OBJ_SUFFIX = o
o = $(OBJ_SUFFIX)

DIRS=BIGGUY   LOCALCUT COMBS    CUT      EDGEGEN  FMATCH   \
     HELDKARP KDTREE   LINKERN  LP       PQ       TINY     TOOLS    \
     TSP      UTIL     VERIFY

all: concorde.h concorde.a
	for i in $(DIRS); do ( cd $$i; $(MAKE) $@ ); done

everything: all
	for i in $(DIRS); do ( cd $$i; $(MAKE) $@ ); done

.PHONY: concorde.a
concorde.a:
	for i in $(DIRS); do ( cd $$i; $(MAKE) concorde.a ); done

INC_LIST=$(BLDINCDIR)/config.h  $(CCINCDIR)/machdefs.h $(CCINCDIR)/util.h     \
         $(CCINCDIR)/bigguy.h   $(CCINCDIR)/combs.h    $(CCINCDIR)/cut.h      \
         $(CCINCDIR)/delaunay.h $(CCINCDIR)/edgegen.h  $(CCINCDIR)/fmatch.h   \
         $(CCINCDIR)/heldkarp.h $(CCINCDIR)/kdtree.h   $(CCINCDIR)/linkern.h  \
         $(CCINCDIR)/lp.h       $(CCINCDIR)/tsp.h      $(CCINCDIR)/consec1.h  \
         $(CCINCDIR)/localcut.h $(CCINCDIR)/macrorus.h $(CCINCDIR)/mlinkern.h \
         $(CCINCDIR)/necklace.h $(CCINCDIR)/pqsets.h   $(CCINCDIR)/pq.h       \
         $(CCINCDIR)/cuttree.h                                                \
         $(CCINCDIR)/tinytsp.h  $(CCINCDIR)/tsp.h      $(CCINCDIR)/verify.h

concorde.h: $(INC_LIST) Makefile
	cat $(INC_LIST) | grep -v '#include "' > concorde.h

clean:
	-rm -f *.$o concorde.h concorde.a
	for i in $(DIRS); do ( cd $$i; $(MAKE) clean ); done

distclean: clean
	-rm -f Makefile */Makefile INCLUDE/config.h INCLUDE/Makefile.common \
               config.log config.cache config.status

depend.in:
	for i in $(DIRS); do (cd $$i; $(MAKE) depend.in); done
