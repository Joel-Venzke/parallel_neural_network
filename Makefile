all: 

NC=NVCC
NCFLAGS=-O3 -arch=sm_30
CC=gcc
CCFLAGS=
GP=gnuplot
GPFLAGS=

test: 
	./parallel
	./serial

parallel.dat: 
	./parallel

serial.dat:
	./serial

parallel:
	$(NC) $(NCFLAGS) parallel.cu -o parallel

serial:
	$(CC) $(CCFLAGS) serial.cpp -o serial

plot: parallel.dat serial.dat
	$(GP) $(GPFLAGS) plot.gp

clean:
	rm serial parallel *.jpg