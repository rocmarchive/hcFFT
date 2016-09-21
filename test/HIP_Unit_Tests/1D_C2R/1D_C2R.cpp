#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <hip_runtime.h>
#include <hipfft.h>

#define NX 256
#define BATCH 1

int main()
{
	hipfftHandle plan;
	int Csize = (NX/2) + 1;
	hipfftComplex *input = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));
	cufftReal *output = (cufftReal*)calloc(NX, sizeof(cufftReal));
	hipfftComplex *idata;
	cufftReal *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Csize; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	hipMalloc((void**)&idata, sizeof(hipfftComplex)*Csize*BATCH);
	hipMemcpy(idata, input, sizeof(hipfftComplex)*Csize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(cufftReal)*NX*BATCH);
	hipMemcpy(odata, output, sizeof(cufftReal)*NX*BATCH, hipMemcpyHostToDevice);
	
	if (hipGetLastError() != hipSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;	
	}

	if (hipfftPlan1d(&plan, NX, HIPFFT_C2R, BATCH) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;	
	}	

	if (hipfftExecC2R(plan, (hipfftComplex*)idata, (cufftReal*)odata) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2R failed");
		return 0;	
	}

	if (hipDeviceSynchronize() != hipSuccess){
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

