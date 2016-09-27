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
	hipfftComplex *input = (hipfftComplex*)calloc(NX, sizeof(hipfftComplex));
	hipfftComplex *output = (hipfftComplex*)calloc(NX, sizeof(hipfftComplex));
	hipfftComplex *idata;
	hipfftComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	hipMalloc((void**)&idata, sizeof(hipfftComplex)*NX*BATCH);
	hipMemcpy(idata, input, sizeof(hipfftComplex)*NX*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(hipfftComplex)*NX*BATCH);
	hipMemcpy(odata, output, sizeof(hipfftComplex)*NX*BATCH, hipMemcpyHostToDevice);

	if (hipGetLastError() != hipSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;
	}

	if (hipfftPlan1d(&plan, NX, HIPFFT_C2C, BATCH) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;
	}

	if (hipfftExecC2C(plan, (hipfftComplex*)idata, (hipfftComplex *)odata, HIPFFT_FORWARD) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return 0;
	}

	if (hipfftExecC2C(plan,(hipfftComplex*) idata, (hipfftComplex *)odata, HIPFFT_INVERSE) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
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

