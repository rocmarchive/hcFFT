#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <hip_runtime.h>
#include <hipfft.h>

#define NX 64
#define NY 128
#define NZ 128
#define BATCH 1

int main()
{
	hipfftHandle plan;
	int Csize = NX * NY * NZ;
	hipfftComplex *input = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));
	hipfftComplex *output = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));
	hipfftComplex *idata;
	hipfftComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Csize; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	hipMalloc((void**)&idata, sizeof(hipfftComplex)*Csize*BATCH);
	hipMemcpy(idata, input, sizeof(hipfftComplex)*Csize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(hipfftComplex)*Csize*BATCH);
	hipMemcpy(odata, output, sizeof(hipfftComplex)*Csize*BATCH, hipMemcpyHostToDevice);

	if (hipGetLastError() != hipSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;
	}

	/* Create a 3D FFT plan. */
	if (hipfftPlan3d(&plan, NX, NY, NZ, HIPFFT_C2C) != HIPFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;
	}

	/* Use the CUFFT plan to transform the signal in place. */
	if (hipfftExecC2C(plan, (hipfftComplex*)idata, (hipfftComplex*)odata, HIPFFT_FORWARD) != HIPFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return 0;
	}
	if (hipfftExecC2C(plan, (hipfftComplex*)idata, (hipfftComplex*)odata, HIPFFT_INVERSE) != HIPFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
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

