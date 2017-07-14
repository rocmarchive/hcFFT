/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef TEST_UNIT_API_HCFFT_TRANSFORMS_HELPER_FUNCTIONS_H_
#define TEST_UNIT_API_HCFFT_TRANSFORMS_HELPER_FUNCTIONS_H_
#include <math.h>
#include <stdlib.h>

double rmse_tolerance = 0.00002;
const double magnitude_lower_limit = 1.0E-100;

// RMSE accuragcy judegement
template <typename T, typename S>
inline bool JudgeRMSEAccuracyComplex(T* expected, S* actual,
                                     unsigned int problem_size_per_transform) {
  double rmse_tolerance_this =
      rmse_tolerance * sqrt(static_cast<double>(problem_size_per_transform) / 4096.0);
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
    mag = ex_r * ex_r + ex_i * ex_i;
    maxMag = (mag > maxMag) ? mag : maxMag;
    // compute square error
    rms += ((ex_r - ac_r) * (ex_r - ac_r) + (ex_i - ac_i) * (ex_i - ac_i));
  }
  if (maxMag > magnitude_lower_limit) {
    maxMagInv = 1.0 / maxMag;
  }
  rms = sqrt(rms * maxMagInv);
  if (fabs(rms) > rmse_tolerance_this) {
    std::cout << std::endl
              << "RMSE accuracy judgement failure -- RMSE = " << std::dec << rms
              << ", maximum allowed RMSE = " << std::dec << rmse_tolerance_this
              << std::endl;
    return 1;
  }
  return 0;
}

// For Real Outputs

// RMSE accurgcy judegement
template <typename T, typename S>
inline bool JudgeRMSEAccuracyReal(T* expected, S* actual,
                                  unsigned int problem_size_per_transform) {
  double rmse_tolerance_this =
      rmse_tolerance * sqrt(static_cast<double>(problem_size_per_transform) / 4096.0);
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
    mag = ex_r * ex_r + ex_i * ex_i;
    maxMag = (mag > maxMag) ? mag : maxMag;
    // compute square error
    rms += ((ex_r - ac_r) * (ex_r - ac_r) + (ex_i - ac_i) * (ex_i - ac_i));
  }
  if (maxMag > magnitude_lower_limit) {
    maxMagInv = 1.0 / maxMag;
  }
  rms = sqrt(rms * maxMagInv);
  if (fabs(rms) > rmse_tolerance_this) {
    std::cout << std::endl
              << "RMSE accuracy judgement failure -- RMSE = " << std::dec << rms
              << ", maximum allowed RMSE = " << std::dec << rmse_tolerance_this
              << std::endl;
    return 1;
  }
  return 0;
}

#endif  // TEST_UNIT_API_HCFFT_TRANSFORMS_HELPER_FUNCTIONS_H_

