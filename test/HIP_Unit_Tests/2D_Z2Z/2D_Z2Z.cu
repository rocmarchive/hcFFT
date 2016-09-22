#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 256 
#define NY 128  
#define BATCH 1

int main()
{
	cufftHandle plan;
	int Csize = ((NX/2) + 1) * NY;
	cufftDoubleComplex *input = (cufftDoubleComplex*)calloc(Csize, sizeof(cufftDoubleComplex));
	cufftDoubleComplex *output = (cufftDoubleComplex*)calloc(Csize, sizeof(cufftDoubleComplex));
	cufftDoubleComplex *idata;
	cufftDoubleComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Csize; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftDoubleComplex)*Csize*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftDoubleComplex)*Csize*BATCH, cudaMemcpyHostToDevice);

	if (cudaGetLastError() != cudaSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 0;
	} 

	/* Create a 2D FFT plan. */ 
	if (cufftPlan2d(&plan, NX, NY, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
		return 0;
	}

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return 0;	
	}

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
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

