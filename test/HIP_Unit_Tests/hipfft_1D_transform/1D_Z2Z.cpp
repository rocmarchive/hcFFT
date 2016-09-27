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
	hipfftDoubleComplex *input = (hipfftDoubleComplex*)calloc(NX, sizeof(hipfftDoubleComplex));
	hipfftDoubleComplex *output = (hipfftDoubleComplex*)calloc(NX, sizeof(hipfftDoubleComplex));
	hipfftDoubleComplex *idata;
	hipfftDoubleComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	hipMalloc((void**)&idata, sizeof(hipfftDoubleComplex)*NX*BATCH);
	hipMemcpy(idata, input, sizeof(hipfftDoubleComplex)*NX*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(hipfftDoubleComplex)*NX*BATCH);
	hipMemcpy(odata, output, sizeof(hipfftDoubleComplex)*NX*BATCH, hipMemcpyHostToDevice);

	if (hipGetLastError() != hipSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;
	}

	if (hipfftPlan1d(&plan, NX, HIPFFT_Z2Z, BATCH) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;
	}

	if (hipfftExecZ2Z(plan, (hipfftDoubleComplex*)idata, (hipfftDoubleComplex*)odata, HIPFFT_FORWARD) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return 0;
	}

	if (hipfftExecZ2Z(plan, (hipfftDoubleComplex*)idata, (hipfftDoubleComplex*)odata, HIPFFT_INVERSE) != HIPFFT_SUCCESS){
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

