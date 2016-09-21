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
	int Rsize = NX * NY * NZ;
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

	/* Create a 3D FFT plan. */ 
	if (hipfftPlan3d(&plan, NX, NY, NZ, HIPFFT_R2C) != HIPFFT_SUCCESS) 
	{ 
		fprintf(stderr, "CUFFT error: Plan creation failed"); 
		return 0;	
	}	

	/* Use the CUFFT plan to transform the signal in place. */ 
	if (hipfftExecR2C(plan, (cufftReal*)idata, (hipfftComplex*)odata) != HIPFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT error: ExecR2C failed"); 
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

