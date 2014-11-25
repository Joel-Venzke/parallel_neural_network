all: plot

NC=nvcc
NCFLAGS=-O3 -arch=sm_30
CC=gcc
CCFLAGS=
GP=gnuplot
GPFLAGS=

test: parallel serial
	./parallel
	./serial

parallel.dat: parallel
	./parallel

serial.dat: serial
	./serial

parallel: parallel.cu
	$(NC) $(NCFLAGS) parallel.cu -o parallel

serial: serial.cpp
	$(CC) $(CCFLAGS) serial.cpp -o serial

plot: parallel.dat serial.dat plot.gp
	$(GP) $(GPFLAGS) plot.gp

clean:
	rm serial parallel *.jpg