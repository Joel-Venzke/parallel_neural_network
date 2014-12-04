#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZE 900
#define HIDDINLAYERS 2
#define POINTS 583
#define TEST 100
#define ATTRIBUTES 10
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ __host__ float g(float value) {
	return 1.0/(1.0+exp(-value));
}

__device__ __host__ float gSquash(float value) {
	if (value<0.5) return 0.0;
	else return 1.0;
}

__device__ __host__ float gPrime(float value){
	float a = g(value);
	return a;
}
__host__ void gLayer2(float *weights, float *values, float *outputs, float *in, float *bias, int weightsLen, int outputsLen){
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
__global__ void gLayerBack(float *weights, float *delta, float *outputs, float *in, int inputLen, int outputsLen){
	float temp;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	while(i < outputsLen)
	{
		temp=0.0;
		for (int j = 0; j < inputLen; ++j)
		{
			temp+=delta[j]*weights[j+SIZE*i];
		}
		
		outputs[i] = gPrime(in[i])*temp;
		i += gridDim.x*blockDim.x;
	}
}

__global__ void gLayer(float *weights, float *values, float *outputs, float *in, float *bias, int weightsLen, int outputsLen){
	float temp;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	while(i < outputsLen) {
		temp=0;
		for (int j = 0; j < weightsLen; ++j)
		{
			temp+= values[j]*weights[j+SIZE*i];
		}
		in[i] = temp+bias[i];
		outputs[i] =g(in[i]);
		i += gridDim.x*blockDim.x;
	}
}

__global__ void updateWeight(float *weights, float *values, float *delta, float learningRate, int inputLen, int outputLen){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	while(i < outputLen)
	{
		while (j < inputLen)
		{
			weights[j+SIZE*i]+= learningRate*values[i]*delta[j];
			j += gridDim.y*blockDim.y;
		}
		j = threadIdx.y + blockDim.y * blockIdx.y;
		i += gridDim.x*blockDim.x;
	}
}

__global__ void deltaInit(float *delta, float *in, float *dataSet, float *values){
	delta[0] = gPrime(in[0])*(dataSet[ATTRIBUTES-1]-values[0]);
}

__global__ void valueInit(float *values, float *dataSet, int i){
	values[i] = dataSet[i];
}

int main(int argc, char const *argv[])
{
	FILE *fp=NULL, *inFile=NULL;
	clock_t t;
	srand (time(NULL));
	float weights[(HIDDINLAYERS+1)][SIZE*SIZE]; // each node in a layer conects to all nodes in the previous layer
	float bias[(HIDDINLAYERS+1)][SIZE]; // all nodes other than the input layer
	float values[(HIDDINLAYERS+2)][SIZE]; // holds the results of the last input
	float in[(HIDDINLAYERS+1)][SIZE]; // values before squashing
	float dataSet[(POINTS+TEST)][ATTRIBUTES]; // holds datafile
	float learningRate = 0.3;
	float learningTime = 50;
	float *weights_d[(HIDDINLAYERS+1)], *bias_d[(HIDDINLAYERS+1)], *values_d[(HIDDINLAYERS+2)], *in_d[(HIDDINLAYERS+1)], *delta_d[(HIDDINLAYERS+1)], *dataSet_d[(POINTS+TEST)];

	// read in data
	inFile=fopen("data/breast-cancer-wisconsin.data","r");
	for (int i = 0; i < POINTS+TEST; ++i)
	{
		for (int j = 0; j < ATTRIBUTES; ++j)
		{
			// update indexes
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
			gLayer2(weights[i], values[i], values[i+1], in[i], bias[i], inputLen, outputLen);
		}
		if (dataSet[j][ATTRIBUTES-1]==gSquash(values[HIDDINLAYERS+1][0])) correct += 1;
	}
	correct = ((float) correct/TEST);
	printf("%f\n", correct);

	// allocate space on device
	for (int i = 0; i < (HIDDINLAYERS+1); ++i)
	{
		HANDLE_ERROR(cudaMalloc((void **) &weights_d[i], sizeof(float)*SIZE*SIZE));
		HANDLE_ERROR(cudaMemcpy(weights_d[i], weights[i], sizeof(float)*SIZE*SIZE, cudaMemcpyHostToDevice));		
		HANDLE_ERROR(cudaMalloc((void **) &bias_d[i], sizeof(float)*SIZE));
		HANDLE_ERROR(cudaMemcpy(bias_d[i], bias[i], sizeof(float)*SIZE, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMalloc((void **) &in_d[i], sizeof(float)*SIZE));
		HANDLE_ERROR(cudaMalloc((void **) &delta_d[i], sizeof(float)*SIZE));
	
	}
	for (int i = 0; i < (HIDDINLAYERS+2); ++i)
	{
		HANDLE_ERROR(cudaMalloc((void **) &values_d[i], sizeof(float)*SIZE));
	}
	for (int i = 0; i < (POINTS+TEST); ++i)
	{
		HANDLE_ERROR(cudaMalloc((void **) &dataSet_d[i], sizeof(float)*ATTRIBUTES));
		HANDLE_ERROR(cudaMemcpy(dataSet_d[i], dataSet[i], sizeof(float)*ATTRIBUTES, cudaMemcpyHostToDevice));
	}

	printf("here\n");
	// initailize kernal launches
	dim3 dimBlock1(32,1);
    dim3 dimGrid1(SIZE/32+1,1);
    dim3 dimBlock2(1,1);
    dim3 dimGrid2(1,1);
    dim3 dimBlock3(16,16);
    dim3 dimGrid3(SIZE/16+1,SIZE/16+1);
    t = clock();
	for (int timeStep=0; timeStep<learningTime; timeStep++) {
		
		for (int point=0; point<POINTS; point++) {

			// get current data set
			for (int i=0; i<ATTRIBUTES-1; i++) {
				valueInit<<<dimGrid2,dimBlock2>>>(values_d[0], dataSet_d[point], i);
				HANDLE_ERROR(cudaGetLastError());
			}

			// forward prop
			for (int i = 0; i < HIDDINLAYERS+1; ++i)
			{
				outputLen=SIZE;
				inputLen=SIZE;
				if (i == HIDDINLAYERS) outputLen = 1; // result layer
				if (i == 0) inputLen = (ATTRIBUTES-1); // data set layer
				gLayer<<<dimGrid1,dimBlock1>>>(weights_d[i], values_d[i], values_d[i+1], in_d[i], bias_d[i], inputLen, outputLen);
				HANDLE_ERROR(cudaGetLastError());
			}

			// back prop 
			deltaInit<<<dimGrid2,dimBlock2>>>(delta_d[HIDDINLAYERS], in_d[HIDDINLAYERS], dataSet_d[point], values_d[HIDDINLAYERS+1]);
			HANDLE_ERROR(cudaGetLastError());

			// error in pervious layers
			
			for (int i = HIDDINLAYERS-1; i > -1; i--)
			{
				outputLen=SIZE;
				inputLen=SIZE;
				if (i == HIDDINLAYERS-1) outputLen = (ATTRIBUTES-1); // data set layer
				if (i == 0) inputLen = 1; // result layer
				gLayerBack<<<dimGrid1,dimBlock1>>>(weights_d[i+1], delta_d[i+1], delta_d[i], in_d[i], inputLen, outputLen);
				HANDLE_ERROR(cudaGetLastError());
			}

			// update weights
			for (int i = 0; i < HIDDINLAYERS+1; ++i)
			{
				// printf("%d\n", i);
				outputLen=SIZE; 
				inputLen=SIZE;
				if (i == HIDDINLAYERS) outputLen = 1; // result layer
				if (i == 0) inputLen = (ATTRIBUTES-1); // data set layer
				updateWeight<<<dimGrid3,dimBlock3>>>(weights_d[i], values_d[i], delta_d[i], learningRate, inputLen, outputLen);
				HANDLE_ERROR(cudaGetLastError());
			}
		}
	}
	t = clock() - t;
	// mem copy 
	for (int i = 0; i < (HIDDINLAYERS+1); ++i)
	{
		HANDLE_ERROR(cudaMemcpy(weights[i], weights_d[i], sizeof(float)*SIZE*SIZE, cudaMemcpyDeviceToHost));
		// HANDLE_ERROR(cudaMemcpy(weights[i], weights_d[i], sizeof(float)*SIZE*SIZE, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(bias[i], bias_d[i], sizeof(float)*SIZE, cudaMemcpyDeviceToHost));
	}

	// free pointers
	for (int i = 0; i < (HIDDINLAYERS+1); ++i)
	{
		HANDLE_ERROR(cudaFree(weights_d[i]));
		HANDLE_ERROR(cudaFree(bias_d[i]));
		HANDLE_ERROR(cudaFree(in_d[i]));
		HANDLE_ERROR(cudaFree(delta_d[i]));	
	}
	for (int i = 0; i < (HIDDINLAYERS+2); ++i)
	{
		HANDLE_ERROR(cudaFree(values_d[i]));
	}
	for (int i = 0; i < (POINTS+TEST); ++i)
	{
		HANDLE_ERROR(cudaFree(dataSet_d[i]));
	}
	
    

    // save runtime
	fp=fopen("data/parallel.dat", "a");
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
			gLayer2(weights[i], values[i], values[i+1], in[i], bias[i], inputLen, outputLen);
		}
		if (dataSet[j][ATTRIBUTES-1]==gSquash(values[HIDDINLAYERS+1][0])) correct += 1;
	}
	correct = ((float) correct/TEST);
	printf("%f\n", correct);
	return 0;
}