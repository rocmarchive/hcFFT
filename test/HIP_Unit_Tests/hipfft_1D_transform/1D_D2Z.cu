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
	cufftDoubleReal *input = (cufftDoubleReal*)calloc(NX, sizeof(cufftDoubleReal));
	cufftDoubleComplex *output = (cufftDoubleComplex*)calloc(Csize, sizeof(cufftDoubleComplex));
	cufftDoubleReal *idata;
	cufftDoubleComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i] = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftDoubleReal)*NX*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftDoubleReal)*NX*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftDoubleComplex)*Csize*BATCH, cudaMemcpyHostToDevice);	

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;	
	}

	if (cufftPlan1d(&plan, NX, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: Plan creation failed"); 	
		return 0;	
	} 	

	/* Use the CUFFT plan to transform the signal in place. */ 
	if (cufftExecD2Z(plan, (cufftDoubleReal*)idata, (cufftDoubleComplex*)odata) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed"); 
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

