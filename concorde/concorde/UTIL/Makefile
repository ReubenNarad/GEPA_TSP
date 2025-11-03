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

THISLIB=util.a
LIBSRCS=allocrus.c bgetopt.c  dheaps_i.c edgelen.c  edgemap.c  edgeutil.c \
        eunion.c   fastread.c getdata.c  safe_io.c  sortrus.c  subdiv.c   \
        urandom.c  util.c     zeit.c     genhash.c  priority.c signal.c

ALLSRCS=$(LIBSRCS)

all: $(THISLIB)

everything: all

clean:
	-rm -f *.$o $(THISLIB)

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

allocrus.$o: allocrus.c $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
bgetopt.$o:  bgetopt.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
dheaps_i.$o: dheaps_i.c $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
edgelen.$o:  edgelen.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/macrorus.h 
edgemap.$o:  edgemap.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/macrorus.h 
edgeutil.$o: edgeutil.c $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
eunion.$o:   eunion.c   $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
fastread.$o: fastread.c $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
genhash.$o:  genhash.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
getdata.$o:  getdata.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/macrorus.h 
priority.$o: priority.c $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
safe_io.$o:  safe_io.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
signal.$o:   signal.c   $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
sortrus.$o:  sortrus.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     \
        $(I)/macrorus.h 
urandom.$o:  urandom.c  $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
util.$o:     util.c     $(I)/machdefs.h $(I2)/config.h  $(I)/macrorus.h \
        $(I)/util.h     
zeit.$o:     zeit.c     $(I)/machdefs.h $(I2)/config.h  $(I)/util.h     
