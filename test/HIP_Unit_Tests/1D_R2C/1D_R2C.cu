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
	cufftReal *input = (cufftReal*)calloc(NX, sizeof(cufftReal));
	cufftComplex *output = (cufftComplex*)calloc(Csize, sizeof(cufftComplex));
	cufftReal *idata;
	cufftComplex *odata;
	
	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i] = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftReal)*NX*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftReal)*NX*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftComplex)*Csize*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftComplex)*Csize*BATCH, cudaMemcpyHostToDevice);	

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;	
	}

	if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: Plan creation failed"); 	
		return 0;	
	} 	

	/* Use the CUFFT plan to transform the signal in place. */ 
	if (cufftExecR2C(plan, (cufftReal*)idata, (cufftComplex*)odata) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecR2C Forward failed"); 
		return 0;	
	} 

	if (cudaDeviceSynchronize() != cudaSuccess)
	{ 
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

