#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H
#include<stdlib.h>
#include<math.h>

double rmse_tolerance = 0.00002;
const double magnitude_lower_limit = 1.0E-100;

//RMSE accuragcy judegement
template<typename T, typename S> 
inline bool JudgeRMSEAccuracyComplex(T* expected, S* actual, unsigned int problem_size_per_transform) {
  double rmse_tolerance_this = rmse_tolerance * sqrt((double)problem_size_per_transform / 4096.0);
  double maxMag = 0.0, maxMagInv = 1.0;
  // Compute RMS error relative to maximum magnitude
  double rms = 0;
  for (size_t z = 0; z < problem_size_per_transform; z++) {
    double ex_r, ex_i, ac_r, ac_i;
    double mag;
    ex_r = expected[z][0];
    ac_r = actual[z].x;
    ex_i = expected[z][1];
    ac_i = actual[z].y;
    // find maximum magnitude
    mag = ex_r*ex_r + ex_i*ex_i;
    maxMag = (mag > maxMag) ? mag : maxMag;
    // compute square error
    rms += ((ex_r - ac_r)*(ex_r - ac_r) + (ex_i - ac_i)*(ex_i - ac_i));
  }
  if (maxMag > magnitude_lower_limit)
  {
    maxMagInv = 1.0 / maxMag;
  }
  rms = sqrt(rms*maxMagInv);
  if (fabs(rms) > rmse_tolerance_this)
  {
    std::cout << std::endl << "RMSE accuracy judgement failure -- RMSE = " << std::dec << rms <<
      ", maximum allowed RMSE = " << std::dec << rmse_tolerance_this << std::endl;
    return 1;
  }
  return 0;
}


// For Real Outputs

//RMSE accurgcy judegement
template<typename T, typename S> 
inline bool JudgeRMSEAccuracyReal(T* expected, S* actual, unsigned int problem_size_per_transform) {
  double rmse_tolerance_this = rmse_tolerance * sqrt((double)problem_size_per_transform / 4096.0);
  double maxMag = 0.0, maxMagInv = 1.0;
  // Compute RMS error relative to maximum magnitude
  double rms = 0;
  for (size_t z = 0; z < problem_size_per_transform; z++) {
    double ex_r, ex_i, ac_r, ac_i;
    double mag;
    ex_r = expected[z];
    ac_r = actual[z];
    ex_i = 0;
    ac_i = 0;
    // find maximum magnitude
    mag = ex_r*ex_r + ex_i*ex_i;
    maxMag = (mag > maxMag) ? mag : maxMag;
    // compute square error
    rms += ((ex_r - ac_r)*(ex_r - ac_r) + (ex_i - ac_i)*(ex_i - ac_i));
  }
  if (maxMag > magnitude_lower_limit)
  {
    maxMagInv = 1.0 / maxMag;
  }
  rms = sqrt(rms*maxMagInv);
  if (fabs(rms) > rmse_tolerance_this)
  {
    std::cout << std::endl << "RMSE accuracy judgement failure -- RMSE = " << std::dec << rms <<
      ", maximum allowed RMSE = " << std::dec << rmse_tolerance_this << std::endl;
    return 1;
  }
  return 0;
}
#endif

