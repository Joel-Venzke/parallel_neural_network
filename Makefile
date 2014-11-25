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

reset: resetParallelData resetSerialData
	
resetParallelData: 
	rm parallel.dat
	echo "# NumberOfElements	time">parallel.dat

resetSerialData: 
	rm serial.dat
	echo "# NumberOfElements	time">serial.dat

clean:
	rm serial parallel