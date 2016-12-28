#include "hipfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "helper_functions.h"
#include "hip/hip_runtime_api.h"

TEST(hipfft_2D_transform_test, func_correct_2D_transform_Z2Z ) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hipfftHandle plan;
  hipfftResult status  = hipfftPlan2d(&plan, N1, N2, HIPFFT_Z2Z);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  int hSize = N1 * N2;
  hipfftDoubleComplex* input = (hipfftDoubleComplex*)calloc(hSize, sizeof(hipfftDoubleComplex));
  hipfftDoubleComplex* output = (hipfftDoubleComplex*)calloc(hSize, sizeof(hipfftDoubleComplex));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < hSize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

  hipfftDoubleComplex* idata; 
  hipfftDoubleComplex* odata;
  hipMalloc(&idata, hSize * sizeof(hipfftDoubleComplex));
  hipMemcpy(idata, input, sizeof(hipfftDoubleComplex) * hSize, hipMemcpyHostToDevice);
  hipMalloc(&odata, hSize * sizeof(hipfftDoubleComplex));
  hipMemcpy(odata, output, sizeof(hipfftDoubleComplex) * hSize, hipMemcpyHostToDevice);
  status = hipfftExecZ2Z(plan, idata, odata, HIPFFT_FORWARD);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(output, odata, sizeof(hipfftDoubleComplex) * hSize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  //FFTW work flow
  // input output arrays
  fftw_complex *fftw_in,*fftw_out;
  fftw_plan p;
  fftw_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hSize);
  fftw_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hSize);
  // Populate inputs
  for(int i = 0; i < hSize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  // 2D forward plan
  p = fftw_plan_dft_2d(N2, N1, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);
  // Execute C2R
  fftw_execute(p);

  // Check RMSE: If fails go for pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftw_complex, hipfftDoubleComplex>(fftw_out, output, hSize)) {
    //Check Real Outputs
    for (int i =0; i < hSize; i++) {
      EXPECT_NEAR(fftw_out[i][0] , output[i].x, 0.1); 
    }
    //Check Imaginary Outputs
    for (int i =0; i < hSize; i++) {
      EXPECT_NEAR(fftw_out[i][1] , output[i].y, 0.1); 
    }
  }
    // Free up resources
  fftw_destroy_plan(p);
  fftw_free(fftw_in); fftw_free(fftw_out);
  free(input);
  free(output);
  hipFree(idata);
  hipFree(odata);
}
