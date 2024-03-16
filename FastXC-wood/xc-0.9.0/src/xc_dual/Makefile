# 为nvcc编译器添加的标志
NVCCFLAG=-G

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
		cuda.xc_dual.o \
		cuda.util.o

OBJS=arguproc.o \
	usage.o \
	sacio.o \
	cal_dist.o \
	gen_ccfpath.o \
	gen_pair_dual.o \
	read_segspec.o \
	read_spec_lst.o \
	util.o
	

BINDIR=../../bin
PROG=$(BINDIR)/spec_xc_dual

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG):$(OBJS) $(CUDAOBJS)
	$(NVCC) $(CFLAG) -o $@ $^ -lm  $(LFLAG)

%.o: %.c
	$(CC) $(CFLAG) -c $<

%.o: %.cu
	$(NVCC) $(CFLAG) $(NVCCFLAG) -c $< 

clean:
	-rm *.o
veryclean:
	-rm $(PROG) *.o 
