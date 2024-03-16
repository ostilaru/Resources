#CC=gcc -Wall -DDEBUG
CC=gcc -Wall

CFLAG=-O2
CFLAG=-O0 -g

# static version 
#
#CFLAG=-O2 -static

NVCC=/usr/local/cuda/bin/nvcc
LFLAG=-L/usr/local/cuda/lib64 -lcufft -lcudart

CUDAOBJS=cuda.main.o \
		cuda.xc_mono.o \
		cuda.util.o

OBJS=sacio.o \
	arguproc.o \
	cal_dist.o \
	read_spec_lst.o \
	read_segspec.o \
	gen_pair_mono.o \
	gen_ccfpath.o \
	usage.o \
	util.o
	

BINDIR=../../bin
PROG=$(BINDIR)/spec_xc_mono

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG):$(OBJS) $(CUDAOBJS)
	$(NVCC) $(CFLAG) -o $@ $^ -lm  $(LFLAG)

%.o: %.c
	$(CC) $(CFLAG) -c $<

%.o: %.cu
	$(NVCC) $(CFLAG) -c $< 

clean:
	-rm *.o
veryclean:
	-rm $(PROG) *.o 
