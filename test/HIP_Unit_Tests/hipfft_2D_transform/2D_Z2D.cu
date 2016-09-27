#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 256 
#define NY 128 
#define NRANK 2 
#define BATCH 1

int main()
{
	cufftHandle plan;
	int Csize = ((NX/2) + 1) * NY;
	int Rsize = NX * NY;
	cufftDoubleComplex *input = (cufftDoubleComplex*)calloc(Csize, sizeof(cufftDoubleComplex));
	cufftDoubleReal *output = (cufftDoubleReal*)calloc(Rsize, sizeof(cufftDoubleReal));
	cufftDoubleComplex *idata;
	cufftDoubleReal *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Csize; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	cudaMalloc((void**)&idata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	cudaMemcpy(idata, input, sizeof(cufftDoubleComplex)*Csize*BATCH, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&odata, sizeof(cufftDoubleReal)*Rsize*BATCH);
	cudaMemcpy(odata, output, sizeof(cufftDoubleReal)*Rsize*BATCH, cudaMemcpyHostToDevice);
	
	if (cudaGetLastError() != cudaSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 0;
	} 

	/* Create a 2D FFT plan. */ 
	if (cufftPlan2d(&plan, NX, NY, CUFFT_Z2D) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
		return 0;
	}


	if (cufftExecZ2D(plan, (cufftDoubleComplex*)idata, (cufftDoubleReal*)odata) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecZ2D failed\n"); 
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

