#include "hcfft_2D_transform.h"
#include "gtest/gtest.h"

  
TEST(hcfft_2D_transform_test, func_correct_2D_transform ) {

  FFTPlan plan;
  hcfftDim dimension = HCFFT_2D;
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftResLocation location = HCFFT_OUTOFPLACE;
  hcfftResTransposed transposeType = HCFFT_NOTRANSPOSE;
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  hcfftPlanHandle planhandle;
  hcfftPrecision precision = HCFFT_SINGLE;
  size_t N1, N2;
  N1 = 2;
  N2 = 12;
  length[0] = N1;
  length[1] = N2;
  int realsize, cmplexsize;
  realsize = length[0] * length[1];
  cmplexsize = length[1] * (1 + (length[0] / 2)) * 2;
  float* input = (float*)calloc(realsize, sizeof(float));
  float* inputz = (float*)calloc(realsize, sizeof(float));
  float* output = (float*)calloc(cmplexsize, sizeof(float));

  for(int  i = 0; i < realsize ; i++) {
    input[i] = i + 1;
  }

  hcfftStatus status = plan.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir);
  status = plan.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set plan error " << endl;
  }

  status = plan.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set result error " << endl;
  }

  Concurrency::array_view<float, 1> inpAr(realsize, input );
  Concurrency::array_view<float, 1> inpAr1(realsize, inputz );
  Concurrency::array_view<float, 1> opAr(cmplexsize, output );
  //---------------------R2C--------------------------------------
  status = plan.hcfftSetLayout(planhandle, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set layout error " << endl;
  }

  status = plan.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCEEDS) {
    cout << " bake plan error " << endl;
  }

  plan.hcfftEnqueueTransform(planhandle, dir, &inpAr, &opAr, NULL);
  opAr.synchronize();
  status = plan.hcfftDestroyPlan(&planhandle);

  
  //---------------------C2R---------------------------------------
  FFTPlan plan1;
  dir = HCFFT_BACKWARD;
  status = plan1.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir);
  status = plan1.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set plan error " << endl;
  }

  status = plan1.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan1.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set result error " << endl;
  }

  status = plan1.hcfftSetLayout(planhandle, HCFFT_HERMITIAN_INTERLEAVED, HCFFT_REAL);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set layout error " << endl;
  }

  status = plan1.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCEEDS) {
    cout << " bake plan error " << endl;
  }

  plan1.hcfftEnqueueTransform(planhandle, dir, &opAr, &inpAr1, NULL);
  inpAr1.synchronize();


  for(int  i = 0; i < realsize; i++) {
    //EXPECT_EQ(round(inpAr1[i]), input[i]);
    //ASSERT_EQ(inpAr1[i], NAN); 
  }

  status = plan1.hcfftDestroyPlan(&planhandle);
}
