CC=gcc -Wall

CFLAG=-O0

OBJS=main.o\
	arguproc.o \
	usage.o\
	sacio.o \
	create_dir.o \
	read_sac_lst.o \
	util.o

BINDIR=../../bin
PROG=$(BINDIR)/ncfstack

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG):$(OBJS)
	$(CC) $(CFLAG) -o $@ $^ -lm 

%.o: %.c
	$(CC) $(CFLAG) -c $< 

 
clean:
	-rm *.o
veryclean:
	-rm $(PROG) *.o 
