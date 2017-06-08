#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "hc_am.hpp"
#include "hcfftlib.h"

TEST(hcfft_3D_transform_test, func_correct_3D_transform_Z2D_RTT) {
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 4;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 4;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 4;
  hcfftHandle plan;
  hcfftResult status  = hcfftPlan3d(&plan, N1, N2, N3, HCFFT_D2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N1 * N2 * N3;
  int Csize = N1 * N2 * (1 + N3 / 2);
  hcfftDoubleReal* inputR2C = (hcfftDoubleReal*)malloc(Rsize * sizeof(hcfftDoubleReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    inputR2C[i] = i%8;
  }

  hcfftDoubleComplex* outputR2C = (hcfftDoubleComplex*)malloc(Csize * sizeof(hcfftDoubleComplex));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftDoubleReal* devIpR2C = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  accl_view.copy(inputR2C, devIpR2C, sizeof(hcfftDoubleReal) * Rsize);
  hcfftDoubleComplex* devOpR2C = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  accl_view.copy(outputR2C, devOpR2C, sizeof(hcfftDoubleComplex) * Csize);
  status = hcfftExecD2Z(plan, devIpR2C, devOpR2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(devOpR2C, outputR2C, sizeof(hcfftDoubleComplex) * Csize);
  status =  hcfftDestroy(plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //plan = NULL;
  status  = hcfftPlan3d(&plan, N1, N2, N3, HCFFT_Z2D);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  hcfftDoubleComplex* inputC2R = (hcfftDoubleComplex*)malloc(Csize * sizeof(hcfftDoubleComplex));
  hcfftDoubleReal* outputC2R = (hcfftDoubleReal*)malloc(Rsize * sizeof(hcfftDoubleReal));

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    inputC2R[i].x = outputR2C[i].x;
    inputC2R[i].y = outputR2C[i].y;
  }

  hcfftDoubleComplex* devIpC2R = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  accl_view.copy(inputC2R, devIpC2R, sizeof(hcfftDoubleComplex) * Csize);
  hcfftDoubleReal* devOpC2R = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  accl_view.copy(outputC2R, devOpC2R, sizeof(hcfftDoubleReal) * Rsize);
  status = hcfftExecZ2D(plan, devIpC2R, devOpC2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(devOpC2R, outputC2R, sizeof(hcfftDoubleReal) * Rsize);
  status =  hcfftDestroy(plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //Check Real Inputs and  Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(inputR2C[i] , outputC2R[i]/Rsize, 0.1); 
  }
  // Free up resources
  free(inputC2R);
  free(outputC2R);
  free(outputR2C);
  free(inputR2C);
  hc::am_free(devIpR2C);
  hc::am_free(devIpC2R);
  hc::am_free(devOpR2C);
  hc::am_free(devOpC2R);
}

