#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZE 200
#define HIDDINLAYERS 2
#define POINTS 583
#define TEST 100
#define ATTRIBUTES 10


float g(float value) {
	return 1.0/(1.0+exp(-value));
}

float gSquash(float value) {
	if (value<0.5) return 0.0;
	else return 1.0;
}

float gPrime(float value){
	float a = g(value);
	return a;
}

void gLayerBack(float *weights, float *delta, float *outputs, float *in, int inputLen, int outputsLen){
	float temp;
	// printf("Here\n");
	for (int i = 0; i < outputsLen; ++i)
	{
		temp=0.0;
		for (int j = 0; j < inputLen; ++j)
		{
			temp+=delta[j]*weights[j+SIZE*i];
		}
		
		outputs[i] = gPrime(in[i])*temp;
		// printf("%f\t%f\t%f\n", temp, in[i], outputs[i]);
	}
}

void gLayer(float *weights, float *values, float *outputs, float *in, float *bias, int weightsLen, int outputsLen){
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

void updateWeight(float *weights, float *values, float *delta, float learningRate, int inputLen, int outputLen){
	for (int i = 0; i < outputLen; ++i)
	{
		for (int j = 0; j < inputLen; ++j)
		{
			weights[j+SIZE*i]+= learningRate*values[i]*delta[j];
			// printf("%f\t%f\n", weights[j+SIZE*i],learningRate*values[i]*delta[j]);
		}
	}
}



int main(int argc, char const *argv[])
{
	FILE *fp=NULL, *inFile=NULL;
	clock_t t;
	srand (time(NULL));
	float weights[HIDDINLAYERS+1][SIZE*SIZE]; // each node in a layer conects to all nodes in the previous layer
	float bias[HIDDINLAYERS+1][SIZE]; // all nodes other than the input layer
	float values[HIDDINLAYERS+2][SIZE]; // holds the results of the last input
	float in[HIDDINLAYERS+1][SIZE]; // values before squashing
	float delta[HIDDINLAYERS+1][SIZE]; // error
	float dataSet[POINTS+TEST][ATTRIBUTES]; // holds datafile
	float learningRate = 0.3;
	float learningTime = 50;

	// read in data
	inFile=fopen("data/breast-cancer-wisconsin.data","r");
	for (int i = 0; i < POINTS+TEST; ++i)
	{
		for (int j = 0; j < ATTRIBUTES; ++j)
		{
			fscanf(inFile,"%f",&(dataSet[i][j]));
			if (j==ATTRIBUTES-1){
				if (dataSet[i][j]==4.0) dataSet[i][j]=1.0;
				else dataSet[i][j] = 0.0;
			} 
		}
	}
	fclose(inFile);


	// initialize weights
	for(int i=0; i<HIDDINLAYERS+1; i++) {
		for (int j = 0; j < SIZE; ++j)
		{
			for (int l = 0; l < SIZE; ++l)
			{
				weights[i][l+SIZE*j] = ((float) rand() / (RAND_MAX/2))-1;
				// printf("%f ", weights[i][l+SIZE*j]);
			}
			// printf("\n");
			bias[i][j] = ((float) rand() / (RAND_MAX/2))-1;
		}	
	}
	int outputLen=SIZE;
	int inputLen=SIZE;
	float correct = 0.0;
	for (int j=POINTS; j<TEST+POINTS; j++){
		for (int i=0; i<ATTRIBUTES-1; i++) {
			values[0][i] = dataSet[j][i];
		}
		for (int i = 0; i < HIDDINLAYERS+1; ++i)
		{
			outputLen=SIZE;
			inputLen=SIZE;
			if (i == HIDDINLAYERS) outputLen = 1; // result layer
			if (i == 0) inputLen = (ATTRIBUTES-1); // data set layer
			gLayer(weights[i], values[i], values[i+1], in[i], bias[i], inputLen, outputLen);
		}
		if (dataSet[j][ATTRIBUTES-1]==gSquash(values[HIDDINLAYERS+1][0])) correct += 1;
	}
	correct = ((float) correct/TEST);
	printf("%f\n", correct);


	t = clock();
	for (int timeStep=0; timeStep<learningTime; timeStep++) {
		// printf("%d\n", timeStep);
		for (int point=0; point<POINTS; point++) {
			

			// get current data set
			for (int i=0; i<ATTRIBUTES-1; i++) {
				values[0][i] = dataSet[point][i];
			}

			// forward prop
			
			for (int i = 0; i < HIDDINLAYERS+1; ++i)
			{
				outputLen=SIZE;
				inputLen=SIZE;
				if (i == HIDDINLAYERS) outputLen = 1; // result layer
				if (i == 0) inputLen = (ATTRIBUTES-1); // data set layer
				gLayer(weights[i], values[i], values[i+1], in[i], bias[i], inputLen, outputLen);
			}

			// back prop
			delta[HIDDINLAYERS][0] = gPrime(in[HIDDINLAYERS][0])*(dataSet[point][ATTRIBUTES-1]-values[HIDDINLAYERS+1][0]); // error in output layer

			// error in pervious layers
			
			for (int i = HIDDINLAYERS-1; i > -1; i--)
			{
				outputLen=SIZE;
				inputLen=SIZE;
				if (i == HIDDINLAYERS-1) outputLen = (ATTRIBUTES-1); // data set layer
				if (i == 0) inputLen = 1; // result layer
				gLayerBack(weights[i+1], delta[i+1], delta[i], in[i], inputLen, outputLen);
			}

			// update weights
			for (int i = 0; i < HIDDINLAYERS+1; ++i)
			{
				outputLen=SIZE;
				inputLen=SIZE;
				if (i == HIDDINLAYERS) outputLen = 1; // result layer
				if (i == 0) inputLen = (ATTRIBUTES-1); // data set layer
				updateWeight(weights[i], values[i], delta[i], learningRate, inputLen, outputLen);
			}
		}
	}
    t = clock() - t;

    // save runtime
	fp=fopen("data/serial.dat", "a");
    fprintf (fp, "%d\t%f\n", SIZE,((float)t)/CLOCKS_PER_SEC);
    fclose(fp);
    correct = 0.0;
	for (int j=POINTS; j<TEST+POINTS; j++){
		for (int i=0; i<ATTRIBUTES-1; i++) {
			values[0][i] = dataSet[j][i];
		}
		for (int i = 0; i < HIDDINLAYERS+1; ++i)
		{
			outputLen=SIZE;
			inputLen=SIZE;
			if (i == HIDDINLAYERS) outputLen = 1; // result layer
			if (i == 0) inputLen = (ATTRIBUTES-1); // data set layer
			gLayer(weights[i], values[i], values[i+1], in[i], bias[i], inputLen, outputLen);
		}
		if (dataSet[j][ATTRIBUTES-1]==gSquash(values[HIDDINLAYERS+1][0])) correct += 1;
	}
	correct = ((float) correct/TEST);
	printf("%f\n", correct);
 //    for(int i=0; i<HIDDINLAYERS+1; i++) {
	// 	for (int j = 0; j < SIZE; ++j)
	// 	{
	// 		for (int l = 0; l < SIZE; ++l)
	// 		{
	// 			printf("%f ", weights[i][l+SIZE*j]);
	// 		}
	// 		printf("\n");
	// 	}	
	// }
	return 0;
}