#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include "hcfftlib.h"
#include <dlfcn.h>
#include <map>
#include <amp.h>
#include <amp_short_vectors.h>
#include <cfloat>
#include <stdio.h>
#include <cmath>

#define PRINT 0

using namespace Concurrency::graphics;
using namespace std;
using namespace Concurrency;

int main(int argc, char* argv[]) {
  FFTPlan plan;
  hcfftDim dimension = HCFFT_2D;
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftResLocation location = HCFFT_OUTOFPLACE;
  hcfftResTransposed transposeType = HCFFT_NOTRANSPOSE;
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  hcfftPlanHandle planhandle;
  hcfftPrecision precision = HCFFT_SINGLE;
  size_t N1, N2;
  N1 = atoi(argv[1]);
  N2 = atoi(argv[2]);
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

  if(status != HCFFT_SUCCESS) {
    cout << " set plan error " << endl;
  }

  status = plan.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCESS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCESS) {
    cout << " set result error " << endl;
  }

  Concurrency::array_view<float, 1> inpAr(realsize, input );
  Concurrency::array_view<float, 1> inpAr1(realsize, inputz );
  Concurrency::array_view<float, 1> opAr(cmplexsize, output );
  /*---------------------R2C--------------------------------------*/
  status = plan.hcfftSetLayout(planhandle, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if(status != HCFFT_SUCCESS) {
    cout << " set layout error " << endl;
  }

  status = plan.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCESS) {
    cout << " bake plan error " << endl;
  }

  plan.hcfftEnqueueTransform(planhandle, dir, &inpAr, &opAr, NULL);
  opAr.synchronize();
  status = plan.hcfftDestroyPlan(&planhandle);
#if PRINT

  /* Print Output */
  for (int i = 0; i < cmplexsize; i++) {
    std::cout << " opAr[" << i << "] " << opAr[i] << std::endl;
  }

#endif
  /*---------------------C2R---------------------------------------*/
  FFTPlan plan1;
  dir = HCFFT_BACKWARD;
  status = plan1.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir);
  status = plan1.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCESS) {
    cout << " set plan error " << endl;
  }

  status = plan1.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCESS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan1.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCESS) {
    cout << " set result error " << endl;
  }

  status = plan1.hcfftSetLayout(planhandle, HCFFT_HERMITIAN_INTERLEAVED, HCFFT_REAL);

  if(status != HCFFT_SUCCESS) {
    cout << " set layout error " << endl;
  }

  status = plan1.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCESS) {
    cout << " bake plan error " << endl;
  }

  plan1.hcfftEnqueueTransform(planhandle, dir, &opAr, &inpAr1, NULL);
  inpAr1.synchronize();
#if PRINT

  /* Print Output */
  for (int i = 0; i < realsize; i++) {
    std::cout << " ipAr[" << i << "] " << inpAr1[i] << std::endl;
  }

#endif

  for(int  i = 0; i < realsize; i++)
    if((round(inpAr1[i]) != input[i]) && isnan(inpAr1[i])) {
      cout << " Mismatch at  " << i << " input " << input[i] << " amp " << round(inpAr1[i]) << endl;
      cout << " TEST FAILED " << std::endl;
      exit(0);
    }

  cout << " TEST PASSED " << std::endl;
  status = plan1.hcfftDestroyPlan(&planhandle);
}
