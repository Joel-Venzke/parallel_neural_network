#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZE 10
#define HIDDINLAYERS 2
#define POINTS 683
#define ATTRIBUTES 10


double g(double value) {
	return 1.0/(1.0+exp(-value));
}

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

void gLayer(double *weights, double *values, double *outputs, double *in, double *bias, int weightsLen, int outputsLen){
	for (int i = 0; i < outputsLen; ++i)
	{
		in[i]=0;
		for (int j = 0; j < weightsLen; ++j)
		{
			in[i]+= values[j]*weights[j+SIZE*i];
		}
		in[i] += bias[i];
		outputs[i] =g(in[i]);
	}
}



double gPrime(double value){
	double a = g(value);
	return a*(1.0-a);
}

void updateWeight(){

}



int main(int argc, char const *argv[])
{
	FILE *fp=NULL, *inFile=NULL;
	inFile=fopen("data/breast-cancer-wisconsin.data","r");
	fp=fopen("data/serial.dat", "a");
	clock_t t;
	srand (time(NULL));
	double weights[HIDDINLAYERS+1][SIZE*SIZE]; // each node in a layer conects to all nodes in the previous layer
	double bias[HIDDINLAYERS+1][SIZE]; // all nodes other than the input layer
	double values[HIDDINLAYERS+2][SIZE]; // holds the results of the last input
	double in[HIDDINLAYERS+1][SIZE]; // values before squashing
	double delta[HIDDINLAYERS+1][SIZE]; // error
	double dataSet[POINTS][ATTRIBUTES]; // holds datafile
	double learningRate = 0.3;
	double learningTime = 50;

	for (int i = 0; i < POINTS; ++i)
	{
		for (int j = 0; j < ATTRIBUTES; ++j)
		{
			fscanf(inFile,"%lf",&(dataSet[i][j]));
			if (j==ATTRIBUTES-1){
				if (dataSet[i][j]==4.0) dataSet[i][j]=1.0;
				else dataSet[i][j] = 0.0;
			} 
			printf("%2.1f ",dataSet[i][j]);
		}
		printf("\n");
	}
	fclose(inFile);


	// initialize weights
	for(int i=0; i<HIDDINLAYERS+1; i++) {
		for (int j = 0; j < SIZE; ++j)
		{
			for (int l = 0; l < SIZE; ++l)
			{
				weights[i][j+SIZE*l] = ((double) rand() / (RAND_MAX/2))-1;
			}
			bias[i][j] = ((double) rand() / (RAND_MAX/2))-1;
		}	
	}

	int outputLen=SIZE;
	int weightsLen=SIZE*SIZE;
	printf("Here\n");

	t = clock();
	for (int timeStep=0; timeStep<learningTime; timeStep++) {
		printf("%d\n", timeStep);
		for (int point=0; point<POINTS; point++) {
			outputLen=SIZE;
			weightsLen=SIZE*SIZE;

			// get current data set
			for (int i=0; i<ATTRIBUTES-1; i++) {
				values[0][i] = dataSet[point][i];
			}

			// forward prop
			for (int i = 0; i < HIDDINLAYERS+1; ++i)
			{
				if (i == HIDDINLAYERS) outputLen = 1; // output layer
				if (i == 0) weightsLen = (ATTRIBUTES-1)*SIZE; // input layer
				gLayer(weights[i], values[i], values[i+1], in[i], bias[i], SIZE*SIZE, outputLen);
			}

			// back prop
			// error in output layer
			delta[HIDDINLAYERS][0] = gPrime(in[HIDDINLAYERS][0])*(dataSet[point][ATTRIBUTES-1]-values[HIDDINLAYERS+1][0]);

			// error in pervious layers
			for (int i = HIDDINLAYERS-1; i < -1; i--)
			{
				
			}

			// update weights
		}
	}
    t = clock() - t;
    fprintf (fp, "%d\t%f\n", SIZE,((float)t)/CLOCKS_PER_SEC);
    fclose(fp);
	return 0;
}