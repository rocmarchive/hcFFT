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

	hipMalloc((void**)&idata, sizeof(cufftDoubleReal)*Rsize*BATCH);
	hipMemcpy(idata, input, sizeof(cufftDoubleReal)*Rsize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	hipMemcpy(odata, output, sizeof(cufftDoubleComplex)*Csize*BATCH, hipMemcpyHostToDevice);	
 
	if (hipGetLastError() != hipSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 0;
	} 

	/* Create a 2D FFT plan. */ 
	if (hipfftPlan2d(&plan, NX, NY, HIPFFT_D2Z) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
		return 0;
	}


	if (hipfftExecD2Z(plan, (cufftDoubleReal*)idata, (cufftDoubleComplex*)odata) != HIPFFT_SUCCESS)
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

