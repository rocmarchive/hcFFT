#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <hip_runtime.h>
#include <hipfft.h>

#define NX 256 
#define NY 128 
#define NRANK 2 
#define BATCH 1

int main()
{
	hipfftHandle plan;
	int Csize = ((NX/2) + 1) * NY;
	int Rsize = NX * NY;
	cufftReal *input = (cufftReal*)calloc(Rsize, sizeof(cufftReal));
	hipfftComplex *output = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));
	cufftReal *idata;
	hipfftComplex *odata;
	
	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Rsize; i++)
	{
		input[i] = rand();
	}

	hipMalloc((void**)&idata, sizeof(cufftReal)*Rsize*BATCH);
	hipMemcpy(idata, input, sizeof(cufftReal)*Rsize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(hipfftComplex)*Csize*BATCH);
	hipMemcpy(odata, output, sizeof(hipfftComplex)*Csize*BATCH, hipMemcpyHostToDevice);	
 
	if (hipGetLastError() != hipSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 0;
	} 

	/* Create a 2D FFT plan. */ 
	if (hipfftPlan2d(&plan, NX, NY, HIPFFT_R2C) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
		return 0;
	}


	if (hipfftExecR2C(plan, (cufftReal*)idata, (hipfftComplex*)odata) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n"); 
		return 0;	
	} 
	if (hipDeviceSynchronize() != hipSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to synchronize\n"); 
		return 0; 
	}

	hipfftDestroy(plan); 

	free(input);
	free(output);

	hipFree(idata);
	hipFree(odata);

	return 0;
}

