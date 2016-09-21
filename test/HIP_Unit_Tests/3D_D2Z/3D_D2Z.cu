#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 128 
#define NY 128 
#define NZ 128 
#define BATCH 1

int main()
{
	cufftHandle plan;
	int Csize = ((NX/2) + 1) * NY * NZ;
	int Rsize = NX * NY * NZ;
	cufftDoubleReal *input = (cufftDoubleReal*)calloc(Rsize, sizeof(cufftDoubleReal));
	cufftDoubleComplex *output = (cufftDoubleComplex*)calloc(Csize, sizeof(cufftDoubleComplex));
	cufftDoubleReal *idata;
	cufftDoubleComplex *odata;
	
	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Rsize; i++)
	{
		input[i] = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftDoubleReal)*Rsize*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftDoubleReal)*Rsize*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftDoubleComplex)*Csize*BATCH, cudaMemcpyHostToDevice);	
 
	if (cudaGetLastError() != cudaSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 0;	
	} 

	/* Create a 3D FFT plan. */ 
	if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS) 
	{ 
		fprintf(stderr, "CUFFT error: Plan creation failed"); 
		return 0;	
	}	

	/* Use the CUFFT plan to transform the signal in place. */ 
	if (cufftExecD2Z(plan, (cufftDoubleReal*)idata, (cufftDoubleComplex*)odata) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecD2Z failed"); 
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

