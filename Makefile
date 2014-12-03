all: src/plot

NC=nvcc
NCFLAGS=-O3 -arch=sm_30
CC=gcc
CCFLAGS=
GP=gnuplot
GPFLAGS=

test: bin/parallel bin/serial
	bin/parallel
	bin/serial

data/parallel.dat: bin/parallel
	bin/parallel

data/serial.dat: bin/serial
	bin/serial

bin/parallel: src/parallel.cu
	$(NC) $(NCFLAGS) src/parallel.cu -o bin/parallel

bin/serial: src/serial.cpp
	$(CC) $(CCFLAGS) src/serial.cpp -o bin/serial

src/plot: data/parallel.dat data/serial.dat src/plot.gp
	$(GP) $(GPFLAGS) src/plot.gp

reset: resetParallelData resetSerialData
	
resetParallelData: 
	rm data/parallel.dat
	echo "# NumberOfElements	time">data/parallel.dat

resetSerialData: 
	rm data/serial.dat
	echo "# NumberOfElements	time">data/serial.dat

clean:
	rm bin/* fit.log