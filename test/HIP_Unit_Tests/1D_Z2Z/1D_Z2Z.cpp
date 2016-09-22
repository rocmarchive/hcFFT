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
	cufftDoubleComplex *input = (cufftDoubleComplex*)calloc(NX, sizeof(cufftDoubleComplex));
	cufftDoubleComplex *output = (cufftDoubleComplex*)calloc(NX, sizeof(cufftDoubleComplex));
	cufftDoubleComplex *idata;
	cufftDoubleComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	hipMalloc((void**)&idata, sizeof(cufftDoubleComplex)*NX*BATCH);
	hipMemcpy(idata, input, sizeof(cufftDoubleComplex)*NX*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(cufftDoubleComplex)*NX*BATCH);
	hipMemcpy(odata, output, sizeof(cufftDoubleComplex)*NX*BATCH, hipMemcpyHostToDevice);
	
	if (hipGetLastError() != hipSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;	
	}

	if (hipfftPlan1d(&plan, NX, HIPFFT_Z2Z, BATCH) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;	
	}	

	if (hipfftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, HIPFFT_FORWARD) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return 0;	
	}

	if (hipfftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, HIPFFT_INVERSE) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecZ2Z Inverse failed");
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

