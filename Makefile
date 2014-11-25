all: 

NC=NVCC
CC=gcc
GP=gnuplot

test: 
	./parallel
	./serial

parallel.cu: 
	$(NC) -O3 -arch=sm_30 parallel.cu -o parallel

serial.cpp: 
	$(CC) serial.cpp -o serial

plot.gp: 
	$(GP) plot.gp