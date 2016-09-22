#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 256
#define BATCH 1

int main()
{
	cufftHandle plan;
	cufftDoubleComplex *input = (cufftDoubleComplex*)calloc(NX, sizeof(cufftDoubleComplex));
	cufftDoubleComplex *output = (cufftDoubleComplex*)calloc(NX, sizeof(cufftDoubleComplex));
	cufftDoubleComplex *idata;
	cufftDoubleComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftDoubleComplex)*NX*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftDoubleComplex)*NX*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftDoubleComplex)*NX*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftDoubleComplex)*NX*BATCH, cudaMemcpyHostToDevice);
	
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;	
	}

	if (cufftPlan1d(&plan, NX, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;	
	}	

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return 0;	
	}

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverse failed");
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

