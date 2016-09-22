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

	hipMalloc((void**)&idata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	hipMemcpy(idata, input, sizeof(cufftDoubleComplex)*Csize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(cufftDoubleReal)*Rsize*BATCH);
	hipMemcpy(odata, output, sizeof(cufftDoubleReal)*Rsize*BATCH, hipMemcpyHostToDevice);
	
	if (hipGetLastError() != hipSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 0;
	} 

	/* Create a 2D FFT plan. */ 
	if (hipfftPlan2d(&plan, NX, NY, HIPFFT_Z2D) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
		return 0;
	}


	if (hipfftExecZ2D(plan, (cufftDoubleComplex*)idata, (cufftDoubleReal*)odata) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecZ2D failed\n"); 
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

