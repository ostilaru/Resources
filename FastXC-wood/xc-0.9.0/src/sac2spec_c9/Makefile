# Compiler and linker options
CC=gcc

CFLAGS=-Wall -O0 -g

NVCC=/usr/local/cuda/bin/nvcc
LFLAG=-L/usr/local/cuda/lib64  -lcufft -lcudart -lm

CUDAOBJS=cuda.alloc_c9.o \
	cuda.onebit.o \
	cuda.rdcrtr.o \
	cuda.smooth.o \
	cuda.util.o \
	cuda.fft_normalize.o \
	cuda.estimatebatch_c9.o \
	cuda.misc.o \
	cuda.preprocess.o \
	cuda.runabs_c9.o \
	cuda.taper.o \
	cuda.whiten_c9.o \
	cuda.main.o \

OBJS=design_filter_response.o \
	arguproc.o \
	cal_nseg.o \
	par_read_sac_c9.o \
	par_write_spec_c9.o \
	read_filelist.o \
	find_whiten_flag.o \
	sacio.o \
	util.o \
	usage.o \
	
BINDIR=../../bin
PROG=$(BINDIR)/sac2spec_c9

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

# Add -G flag for device debuggin info
NVCCFLAGS = -g -G

$(PROG):$(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LFLAG)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -c $<

.PHONY: clean veryclean

clean:
	-rm *.o

veryclean:
	-rm $(PROG) *.o 
