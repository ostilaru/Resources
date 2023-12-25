# Compiler and linker options
CC=gcc
CFLAGS=-Wall -O0 -g

NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS = -g -G
LFLAG=-L/usr/local/cuda/lib64 -lcufft -lcudart -lm

# Object files
CUDAOBJS=cuda.alloc_c1.o \
    cuda.onebit.o \
    cuda.rdcrtr.o \
    cuda.smooth.o \
    cuda.util.o \
    cuda.fft_normalize.o \
    cuda.estimatebatch_c1.o \
    cuda.misc.o \
    cuda.preprocess.o \
    cuda.runabs_c1.o \
    cuda.taper.o \
    cuda.whiten_c1.o \
    cuda.main.o \

OBJS=design_filter_response.o \
    sacio.o \
    arguproc.o \
    cal_nseg.o \
    par_read_sac_c1.o \
    par_write_spec_c1.o \
    read_filelist.o \
    find_whiten_flag.o \
    util.o \
    usage.o \

# Binary directory and program
BINDIR=../../bin
PROG=$(BINDIR)/sac2spec_c1

# Main rule
all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -c $<

# Compile CUDA files
%.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -c $<

# Link objects into a binary
$(PROG):$(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(CUDAOBJS) $(LFLAG)

# Clean up
.PHONY: clean veryclean print-version
clean:
	-rm *.o

veryclean:
	-rm $(PROG) *.o 

# Print compiler and linker version
print-version:
	@echo "GCC version:"
	@gcc --version
	@echo "NVCC version:"
	@$(NVCC) --version
	@echo "Linker version:"
	@ld --version
