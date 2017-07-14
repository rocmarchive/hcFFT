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

#ifndef LIB_SRC_GENERATOR_TRANSPOSE_H_
#define LIB_SRC_GENERATOR_TRANSPOSE_H_

#include "include/stockham.h"
#include <vector>
#define AVAIL_MEM_SIZE 32768

namespace hcfft_transpose_generator {
// generating string for calculating offset within sqaure transpose kernels
// (genTransposeKernelBatched)
void OffsetCalculation(std::stringstream& transKernel,
                       const FFTKernelGenKeyParams& params, bool input);

// generating string for calculating offset within sqaure transpose kernels
// (genTransposeKernelLeadingDimensionBatched)
void OffsetCalcLeadingDimensionBatched(std::stringstream& transKernel,
                                       const FFTKernelGenKeyParams& params);

// generating string for calculating offset within swap kernels (genSwapKernel)
void Swap_OffsetCalc(std::stringstream& transKernel,
                     const FFTKernelGenKeyParams& params);

// Small snippet of code that multiplies the twiddle factors into the
// butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the
// transpose
hcfftStatus genTwiddleMath(const hcfftPlanHandle plHandle,
                           const FFTKernelGenKeyParams& params,
                           std::stringstream& transKernel,
                           const std::string& dtComplex, bool fwd);

// Small snippet of code that multiplies the twiddle factors into the
// butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the
// transpose
hcfftStatus genTwiddleMathLeadingDimensionBatched(
    const hcfftPlanHandle plHandle, const FFTKernelGenKeyParams& params,
    std::stringstream& transKernel, const std::string& dtComplex, bool fwd);

hcfftStatus genTransposePrototype(
    const FFTKernelGenKeyParams& params, const size_t& lwSize,
    const std::string& dtPlanar, const std::string& dtComplex,
    const std::string& funcName, std::stringstream& transKernel,
    std::string& dtInput, std::string& dtOutput, bool twiddleTransposeKernel);

/* -> get_cycles function gets the swapping logic required for given row x col
matrix.
-> cycle_map[0] holds the total number of cycles required.
-> cycles start and end with the same index, hence we can identify individual
cycles,
though we tend to store the cycle index contiguously*/
void get_cycles(size_t* cycle_map, size_t num_reduced_row,
                size_t num_reduced_col);

/*
calculate the permutation cycles consumed in swap kernels.
each cycle is strored in a vecotor. hopfully there are mutliple independent
vectors thus we use a vector of vecotor
*/
void permutation_calculation(size_t m, size_t n,
                             std::vector<std::vector<size_t> >& permutationVec);

// swap lines. This kind of kernels are using with combination of square
// transpose kernels to perform nonsqaure transpose
// this function assumes a 1:2 ratio
hcfftStatus genSwapKernel(const FFTKernelGenKeyParams& params,
                          std::string& strKernel, std::string& KernelFuncName,
                          const size_t& lwSize, const size_t reShapeFactor,
                          std::vector<size_t> gWorkSize,
                          std::vector<size_t> lWorkSize, size_t count);

// swap lines. a more general kernel generator.
// this function accepts any ratio in theory. But in practice we restrict it to
// 1:2, 1:3, 1:5 and 1:10 ration
hcfftStatus genSwapKernelGeneral(
    void** twiddleslarge, hc::accelerator acc, const hcfftPlanHandle plHandle,
    const FFTKernelGenKeyParams& params, std::string& strKernel,
    std::string& KernelFuncName, const size_t& lwSize,
    const size_t reShapeFactor, std::vector<size_t> gWorkSize,
    std::vector<size_t> lWorkSize, size_t count);

// generate transepose kernel with sqaure 2d matrix of row major with arbitrary
// batch size
/*
Below is a matrix(row major) containing three sqaure sub matrix along column
The transpose will be done within each sub matrix.
[M0
 M1
 M2]
*/
hcfftStatus genTransposeKernelBatched(
    void** twiddleslarge, hc::accelerator acc, const hcfftPlanHandle plHandle,
    const FFTKernelGenKeyParams& params, std::string& strKernel,
    const size_t& lwSize, const size_t reShapeFactor,
    std::vector<size_t> gWorkSize, std::vector<size_t> lWorkSize, size_t count);

// generate transpose kernel with square 2d matrix of row major with blocks
// along the leading dimension
// aka leading dimension batched
/*
Below is a matrix(row major) contaning three square sub matrix along row
[M0 M2 M2]
*/
hcfftStatus genTransposeKernelLeadingDimensionBatched(
    void** twiddleslarge, hc::accelerator acc, const hcfftPlanHandle plHandle,
    const FFTKernelGenKeyParams& params, std::string& strKernel,
    const size_t& lwSize, const size_t reShapeFactor,
    std::vector<size_t> gWorkSize, std::vector<size_t> lWorkSize, size_t count);

}  // end of namespace hcfft_transpose_generator

#endif  // LIB_SRC_GENERATOR_TRANSPOSE_H_
