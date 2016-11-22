#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"

TEST(hcfft_3D_transform_test, func_correct_3D_transform_C2R_RTT) {
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 2;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 2;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 2;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan3d(plan, N1, N2, N3, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N3 * N2 * N1;
  int Csize = N3 * N2 * (1 + N1 / 2);
  hcfftReal* inputR2C = (hcfftReal*)malloc(Rsize * sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    inputR2C[i] = i%8;
  }

  hcfftComplex* outputR2C = (hcfftComplex*)malloc(Csize *sizeof(hcfftComplex));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftReal* devIpR2C = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(inputR2C, devIpR2C, sizeof(hcfftReal) * Rsize);
  hcfftComplex* devOpR2C = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(outputR2C, devOpR2C, sizeof(hcfftComplex) * Csize);
  status = hcfftExecR2C(*plan, devIpR2C, devOpR2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(devOpR2C, outputR2C, sizeof(hcfftComplex) * Csize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  plan = NULL;
  status  = hcfftPlan3d(plan, N1, N2, N3, HCFFT_C2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  hcfftComplex* inputC2R = (hcfftComplex*)malloc(Csize * sizeof(hcfftComplex));
  hcfftReal* outputC2R = (hcfftReal*)malloc(Rsize * sizeof(hcfftReal));

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    inputC2R[i].x = outputR2C[i].x;
    inputC2R[i].y = outputR2C[i].y;
  }

  hcfftComplex* devIpC2R = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(inputC2R, devIpC2R, sizeof(hcfftComplex) * Csize);
  hcfftReal* devOpC2R = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(outputC2R, devOpC2R, sizeof(hcfftReal) * Rsize);
  status = hcfftExecC2R(*plan, devIpC2R, devOpC2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(devOpC2R, outputC2R, sizeof(hcfftReal) * Rsize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //Check Real Inputs and  Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(inputR2C[i] , outputC2R[i]/Rsize, 0.1); 
  }
  // Free up resources
  free(inputC2R);
  free(outputC2R);
  free(inputR2C);
  free(outputR2C);
  hc::am_free(devIpC2R);
  hc::am_free(devOpC2R);
  hc::am_free(devIpR2C);
  hc::am_free(devOpR2C);
}

