#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <hip_runtime.h>
#include <hipfft.h>

#define NX 128 
#define NY 128 
#define NZ 128 
#define BATCH 1

int main()
{
	hipfftHandle plan;
	int Csize = ((NX/2) + 1) * NY * NZ;
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

	hipMalloc((void**)&idata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	hipMemcpy(idata, input, sizeof(cufftDoubleComplex)*Csize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(cufftDoubleComplex)*Csize*BATCH);
	hipMemcpy(odata, output, sizeof(cufftDoubleComplex)*Csize*BATCH, hipMemcpyHostToDevice);

	if (hipGetLastError() != hipSuccess)
	{ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 0;	
	} 

	/* Create a 3D FFT plan. */ 
	if (hipfftPlan3d(&plan, NX, NY, NZ, HIPFFT_Z2Z) != HIPFFT_SUCCESS) 
	{ 
		fprintf(stderr, "CUFFT error: Plan creation failed"); 
		return 0;	
	}	

	/* Use the CUFFT plan to transform the signal in place. */ 
	if (hipfftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, HIPFFT_FORWARD) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed"); 
		return 0;	
	}
	if (hipfftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, HIPFFT_INVERSE) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverse failed"); 
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

