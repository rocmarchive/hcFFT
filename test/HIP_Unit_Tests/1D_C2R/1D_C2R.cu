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
	int Csize = (NX/2) + 1;
	cufftComplex *input = (cufftComplex*)calloc(Csize, sizeof(cufftComplex));
	cufftReal *output = (cufftReal*)calloc(NX, sizeof(cufftReal));
	cufftComplex *idata;
	cufftReal *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Csize; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftComplex)*Csize*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftComplex)*Csize*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftReal)*NX*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftReal)*NX*BATCH, cudaMemcpyHostToDevice);
	
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;	
	}

	if (cufftPlan1d(&plan, NX, CUFFT_C2R, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;	
	}	

	if (cufftExecC2R(plan, (cufftComplex*)idata, (cufftReal*)odata) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2R failed");
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

