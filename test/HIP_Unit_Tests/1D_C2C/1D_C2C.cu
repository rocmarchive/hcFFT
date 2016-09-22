#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 16
#define BATCH 1

int main()
{
	cufftHandle plan;
	cufftComplex *input = (cufftComplex*)calloc(NX, sizeof(cufftComplex));
	cufftComplex *output = (cufftComplex*)calloc(NX, sizeof(cufftComplex));
	cufftComplex *idata;
	cufftComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftComplex)*NX*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftComplex)*NX*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyHostToDevice);
	

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;	
	}

	if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;	
	}	

	if (cufftExecC2C(plan, (cufftComplex*)idata, (cufftComplex *)odata, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return 0;	
	}

	if (cufftExecC2C(plan,(cufftComplex*) idata, (cufftComplex *)odata, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
		return 0;	
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return 0;	
	}	

	cufftDestroy(plan);

	free(input);
	free(output);

	cudaFree(idata);
	cudaFree(odata);

	return 0;
}

