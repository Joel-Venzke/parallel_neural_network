#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZE 10
#define LAYERS 2


void gLayerBack(double *weights, double *values, double *outputs, int weightsLen, int outputsLen){
	for (int i = 0; i < outputsLen; ++i)
	{
		outputs[i] = 0;
		for (int j = 0; j < weightsLen; ++j)
		{
			outputs[i]+=g(values[j]*weights[j]);
		}
	}
}

void gLayer(double *weights, double *values, double *outputs, int *in, int weightsLen, int outputsLen){
	for (int i = 0; i < outputsLen; ++i)
	{
		outputs[i] = 0;
		for (int j = 0; j < weightsLen; ++j)
		{
			in[i] = values[j]*weights[j];
			outputs[i]+=g(in[i]);
		}
	}
}

double g(double value) {
	return 1.0/(1.0+exp(-value));
}

double gPrime(double value){
	double a = g(value);
	return a*(1.0-a);
}

void updateWeight(){

}



int main(int argc, char const *argv[])
{
	FILE *fp;
	fp=fopen("data/serial.dat", "a");
	clock_t t;
	t = clock();
	srand (time(NULL));
	double weights[LAYERS][SIZE];
	double values[LAYERS][SIZE];
	double in[LAYERS][SIZE];
	double deltaj;

	// initialize weights
	for(int i=0; i<LAYERS; i++) {
		for (int j = 0; j < SIZE; ++j)
		{
			weights[i][j] = ((double) rand() / (RAND_MAX/2))-1
		}	
	}


	// forward prop
	double in;
	int outputLen=SIZE;
	for (int i = 1; i < LAYERS; ++i)
	{
		if (i = LAYERS-1) outputsLen =1;
 		gLayer(weights[i], value[i], values[i+1], in[i], SIZE, outputsLen);
	}

	// back prop
	deltaj = gPrime(in[LAYERS-1][0])(data[i]); // loop over all values in data set


	// update weights

    t = clock() - t;
    fprintf (fp, "%d\t%f\n", SIZE,((float)t)/CLOCKS_PER_SEC);
    fclose(fp);
	return 0;
}