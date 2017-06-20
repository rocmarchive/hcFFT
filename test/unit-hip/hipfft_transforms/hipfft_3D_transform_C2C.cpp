#include "hipfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "helper_functions.h"
#include "hip/hip_runtime.h"

TEST(hipfft_3D_transform_test, func_correct_3D_transform_C2C ) {
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 2;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 2;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 2;

  // HIPFFT work flow
  hipfftHandle plan;
  hipfftResult status  = hipfftPlan3d(&plan, N1, N2, N3, HIPFFT_C2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  int hSize = N1 * N2 * N3;
  hipfftComplex* input = (hipfftComplex*)malloc(hSize * sizeof(hipfftComplex));
  hipfftComplex* output = (hipfftComplex*)malloc(hSize * sizeof(hipfftComplex));

  // Populate the input
  for(int i = 0; i < hSize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

  hipfftComplex* idata; 
  hipfftComplex* odata;
  hipMalloc(&idata, hSize * sizeof(hipfftComplex));
  hipMemcpy(idata, input, sizeof(hipfftComplex) * hSize, hipMemcpyHostToDevice);
  hipMalloc(&odata, hSize * sizeof(hipfftComplex));
  hipMemcpy(odata, output, sizeof(hipfftComplex) * hSize, hipMemcpyHostToDevice);
  status = hipfftExecC2C(plan, idata, odata, HIPFFT_FORWARD);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(output, odata, sizeof(hipfftComplex) * hSize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  //FFTW work flow
  // input output arrays
  fftwf_complex *fftw_in,*fftw_out;
  fftwf_plan p;
  fftw_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * hSize);
  fftw_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * hSize);
  // Populate inputs
  for(int i = 0; i < hSize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  // 3D forward plan
  p = fftwf_plan_dft_3d(N1, N2, N3, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);
  // Execute C2R
  fftwf_execute(p);

  // Check RMSE: If fails go for pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftwf_complex, hipfftComplex>(fftw_out, output, hSize)) {
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
  fftwf_destroy_plan(p);
  fftwf_free(fftw_in); fftwf_free(fftw_out);
  free(input);
  free(output);
  hipFree(idata);
  hipFree(odata);
}
