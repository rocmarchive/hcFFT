#include "hipfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "helper_functions.h"
#include "hip/hip_runtime.h"

TEST(hipfft_3D_transform_test, func_correct_3D_transform_D2Z ) {
//  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 4;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 4;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 4;
  
  // HIPFFT work flow
  hipfftHandle plan;
  hipfftResult status  = hipfftPlan3d(&plan, N1, N2, N3, HIPFFT_D2Z);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  int Rsize = N1 * N2 * N3;
  int Csize = N3 * N2 * (1 + N1 / 2);
  hipfftDoubleReal* input = (hipfftDoubleReal*)malloc(Rsize * sizeof(hipfftDoubleReal));
  hipfftDoubleComplex* output = (hipfftDoubleComplex*)malloc(Csize * sizeof(hipfftDoubleComplex));

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = i%8;
  }

  hipfftDoubleReal* idata; 
  hipfftDoubleComplex* odata;
  hipMalloc(&idata, Rsize * sizeof(hipfftDoubleReal));
  hipMemcpy(idata, input, sizeof(hipfftDoubleReal) * Rsize, hipMemcpyHostToDevice);
  hipMalloc(&odata, Csize * sizeof(hipfftDoubleComplex));
  hipMemcpy(odata, output, sizeof(hipfftDoubleComplex) * Csize, hipMemcpyHostToDevice);
  status = hipfftExecD2Z(plan, idata, odata);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(output, odata, sizeof(hipfftDoubleComplex) * Csize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  //FFTW work flow
  // input output arrays
  double *in; fftw_complex* out;
  fftw_plan p;
  in = (double*) fftw_malloc(sizeof(double) * Rsize);
  // Populate inputs
  for(int i = 0; i < Rsize ; i++) {
    in[i] = input[i];
  }
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Csize);
  // 3D forward plan
  p = fftw_plan_dft_r2c_3d(N1, N2, N3, in, out, FFTW_ESTIMATE | FFTW_R2HC);;
  // Execute R2C
  fftw_execute(p);

  // Check RMSE: If fails move on to pointwise comparison 
  if (JudgeRMSEAccuracyComplex<fftw_complex, hipfftDoubleComplex>(out, output, Csize)) {
    //Check Real Outputs
    for (int i =0; i < Csize; i++) {
      EXPECT_NEAR(out[i][0] , output[i].x, 0.1); 
    }
    //Check Imaginary Outputs
    for (int i =0; i < Csize; i++) {
      EXPECT_NEAR(out[i][1] , output[i].y, 0.1); 
    }
  }

  //Free up resources
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
  free( input );
  free( output );
  hipFree(idata);
  hipFree(odata);
}

