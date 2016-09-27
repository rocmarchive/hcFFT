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
	hipfftDoubleComplex *input = (hipfftDoubleComplex*)calloc(Csize, sizeof(hipfftDoubleComplex));
	hipfftDoubleReal *output = (hipfftDoubleReal*)calloc(NX, sizeof(hipfftDoubleReal));
	hipfftDoubleComplex *idata;
	hipfftDoubleReal *odata;

	int seed = 123456789;
	srand(seed);

	for(int i=0; i<Csize; i++)
	{
		input[i].x = rand();
		input[i].y = rand();
	}

	hipMalloc((void**)&idata, sizeof(hipfftDoubleComplex)*Csize*BATCH);
	hipMemcpy(idata, input, sizeof(hipfftDoubleComplex)*Csize*BATCH, hipMemcpyHostToDevice);
	hipMalloc((void**)&odata, sizeof(hipfftDoubleReal)*NX*BATCH);
	hipMemcpy(odata, output, sizeof(hipfftDoubleReal)*NX*BATCH, hipMemcpyHostToDevice);

	if (hipGetLastError() != hipSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;
	}

	if (hipfftPlan1d(&plan, NX, HIPFFT_Z2D, BATCH) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;
	}

	if (hipfftExecZ2D(plan, (hipfftDoubleComplex*)idata, (hipfftDoubleReal*)odata) != HIPFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecZ2D failed");
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

