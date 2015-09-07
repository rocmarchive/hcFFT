#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include "../ampfftlib.h"
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

int main(int argc,char* argv[])
{
  FFTPlan plan;
  ampfftDim dimension = AMPFFT_2D;
  ampfftDirection dir = AMPFFT_FORWARD;
  ampfftResLocation location = AMPFFT_OUTOFPLACE;
  ampfftResTransposed transposeType = AMPFFT_NOTRANSPOSE;
  size_t *length = (size_t*)malloc(sizeof(size_t) * dimension);
  ampfftPlanHandle planhandle;
  ampfftPrecision precision = AMPFFT_SINGLE;
  size_t N1, N2;

  N1 = atoi(argv[1]);
  N2 = atoi(argv[2]);

  length[0] = N1;
  length[1] = N2;
  int realsize, cmplexsize;
  realsize = length[0] * length[1];
  cmplexsize = length[1] * (1+(length[0]/2)) * 2;

  float* input = (float*)calloc(realsize, sizeof(float));
  float* inputz = (float*)calloc(realsize, sizeof(float));
  float* output = (float*)calloc(cmplexsize, sizeof(float)); 
  for(int  i = 0; i < realsize ; i++)
  {
    input[i] = i+1;
  }

  ampfftStatus status = plan.ampfftCreateDefaultPlan (&planhandle, dimension, length);
  status = plan.ampfftSetPlanPrecision(planhandle, precision);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set plan error "<<endl;
  }
  status = plan.ampfftSetPlanTransposeResult(planhandle, transposeType);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set ampfftSetPlanTransposeResult error "<<endl;
  }
  status = plan.ampfftSetResultLocation(planhandle, AMPFFT_OUTOFPLACE);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set result error "<<endl;
  }
  cout<< " ampfftSetResultLocation status "<<status<<endl; 

  cout<<" bake plan "<<status<<endl;
  Concurrency::array_view<float, 1> inpAr(realsize, input );
  Concurrency::array_view<float, 1> inpAr1(realsize, inputz );
  Concurrency::array_view<float, 1> opAr(cmplexsize, output );
  /*---------------------R2C--------------------------------------*/
  status = plan.ampfftSetLayout(planhandle, AMPFFT_REAL, AMPFFT_COMPLEX);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set layout error "<<endl;
  }
  status = plan.ampfftBakePlan(planhandle);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" bake plan error "<<endl;
  }
  
  plan.ampfftEnqueueTransform(planhandle, dir, &inpAr, &opAr, NULL);
  opAr.synchronize();

  status = plan.executePlan(&plan);
  status = plan.ampfftDestroyPlan(&planhandle);

#if PRINT
  /* Print Output */
  for (int i = 0; i < cmplexsize; i++)
    std::cout<<" opAr["<<i<<"] "<<opAr[i]<<std::endl;
#endif

  /*---------------------C2R---------------------------------------*/

  FFTPlan plan1;
  status = plan1.ampfftCreateDefaultPlan (&planhandle, dimension, length);
  status = plan1.ampfftSetPlanPrecision(planhandle, precision);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set plan error "<<endl;
  }
  status = plan1.ampfftSetPlanTransposeResult(planhandle, transposeType);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set ampfftSetPlanTransposeResult error "<<endl;
  }
  status = plan1.ampfftSetResultLocation(planhandle, AMPFFT_OUTOFPLACE);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set result error "<<endl;
  }
  cout<< " ampfftSetResultLocation status "<<status<<endl; 

  dir = AMPFFT_BACKWARD;
  status = plan1.ampfftSetLayout(planhandle, AMPFFT_COMPLEX, AMPFFT_REAL);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" set layout error "<<endl;
  }
  status = plan1.ampfftBakePlan(planhandle);
  if(status != AMPFFT_SUCCESS)
  {
    cout<<" bake plan error "<<endl;
  }

  plan1.ampfftEnqueueTransform(planhandle, dir, &opAr, &inpAr1, NULL);

  inpAr1.synchronize();

#if PRINT
  /* Print Output */
  for (int i = 0; i < realsize; i++)
    std::cout<<" ipAr["<<i<<"] "<<inpAr1[i]<<std::endl;
#endif

  for(int  i =0;i<realsize;i++)
  if(abs(inpAr1[i] - input[i]) > 1.0 )
  { 
    cout<<" Mismatch at  "<<i<<" input "<<input[i]<<" amp "<<inpAr1[i]<<endl;
    exit(0);
  }
  cout<<" TEST PASSED"<<endl;
  status = plan1.executePlan(&plan);
  status = plan1.ampfftDestroyPlan(&planhandle);

 }
