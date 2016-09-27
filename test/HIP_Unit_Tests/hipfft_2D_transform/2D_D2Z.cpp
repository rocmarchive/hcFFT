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
	hipfftDoubleReal *input = (hipfftDoubleReal*)calloc(Rsize, sizeof(hipfftDoubleReal));
	hipfftDoubleComplex *output = (hipfftDoubleComplex*)calloc(Csize, sizeof(hipfftDoubleComplex));
	hipfftDoubleReal *idata;
	hipfftDoubleComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Rsize; i++)
	{
		input[i] = rand();
	}

	hipMalloc((void**)&idata, sizeof(hipfftDoubleReal)*Rsize*BATCH);
	hipMemcpy(idata, input, sizeof(hipfftDoubleReal)*Rsize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(hipfftDoubleComplex)*Csize*BATCH);
	hipMemcpy(odata, output, sizeof(hipfftDoubleComplex)*Csize*BATCH, hipMemcpyHostToDevice);

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


	if (hipfftExecD2Z(plan, (hipfftDoubleReal*)idata, (hipfftDoubleComplex*)odata) != HIPFFT_SUCCESS)
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

