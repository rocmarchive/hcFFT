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
	hipfftDoubleReal *input = (hipfftDoubleReal*)calloc(NX, sizeof(hipfftDoubleReal));
	hipfftDoubleComplex *output = (hipfftDoubleComplex*)calloc(Csize, sizeof(hipfftDoubleComplex));
	hipfftDoubleReal *idata;
	hipfftDoubleComplex *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<NX; i++)
	{
		input[i] = rand();
	}

	hipMalloc((void**)&idata, sizeof(hipfftDoubleReal)*NX*BATCH);
	hipMemcpy(idata, input, sizeof(hipfftDoubleReal)*NX*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(hipfftDoubleComplex)*Csize*BATCH);
	hipMemcpy(odata, output, sizeof(hipfftDoubleComplex)*Csize*BATCH, hipMemcpyHostToDevice);

	if (hipGetLastError() != hipSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;
	}

	if (hipfftPlan1d(&plan, NX, HIPFFT_D2Z, BATCH) != HIPFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;
	}

	/* Use the CUFFT plan to transform the signal in place. */
	if (hipfftExecD2Z(plan, (hipfftDoubleReal*)idata, (hipfftDoubleComplex*)odata) != HIPFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecD2Z Forward failed");
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

