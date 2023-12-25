CC=gcc -Wall
CFLAG=-O2 -g
# static version  need glibc-static library
CFLAG=-O2 -g -static

OBJS=main.o \
	usage.o \
	arguproc.o \
	create_dir.o\
	sacio.o  \
	hddistance.o \
	rotate.o \

BINDIR=../../bin
PROG=$(BINDIR)/RotateNCF

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG):$(OBJS)
	$(CC) $(CFLAG) -o $@ $^ -lm 

#
main.o: main.c sac.h
sacio.o: sacio.c sac.h
hddistance.o: hddistance.c

clean:
	-rm *.o
veryclean:
	-rm $(PROG) *.o 
