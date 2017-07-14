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

#include "./generator.transpose.h"

// using namespace StockhamGenerator;

namespace hcfft_transpose_generator {
// generating string for calculating offset within sqaure transpose kernels
// (genTransposeKernelBatched)
void OffsetCalculation(std::stringstream& transKernel,
                       const FFTKernelGenKeyParams& params, bool input) {
  const size_t* stride = input ? params.fft_inStride : params.fft_outStride;
  std::string offset = input ? "iOffset" : "oOffset";

  StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t " << offset << " = 0;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 3) << "g_index = tidx.tile[0];" << std::endl;

  for (size_t i = params.fft_DataDim - 2; i > 0; i--) {
    StockhamGenerator::hcKernWrite(transKernel, 3) << offset << " += (g_index/numGroupsY_" << i
                                << ")*" << stride[i + 1] << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "g_index = g_index % numGroupsY_" << i << ";"
                                << std::endl;
  }

  StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
}

// generating string for calculating offset within sqaure transpose kernels
// (genTransposeKernelLeadingDimensionBatched)
void OffsetCalcLeadingDimensionBatched(std::stringstream& transKernel,
                                       const FFTKernelGenKeyParams& params) {
  const size_t* stride = params.fft_inStride;
  std::string offset = "iOffset";

  StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t " << offset << " = 0;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 3) << "g_index = tidx.tile[0];" << std::endl;

  for (size_t i = params.fft_DataDim - 2; i > 0; i--) {
    StockhamGenerator::hcKernWrite(transKernel, 3) << offset << " += (g_index/numGroupsY_" << i
                                << ")*" << stride[i + 1] << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "g_index = g_index % numGroupsY_" << i << ";"
                                << std::endl;
  }

  StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
}

// generating string for calculating offset within swap kernels (genSwapKernel)
void Swap_OffsetCalc(std::stringstream& transKernel,
                     const FFTKernelGenKeyParams& params) {
  const size_t* stride = params.fft_inStride;
  std::string offset = "iOffset";

  StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t " << offset << " = 0;" << std::endl;

  for (size_t i = params.fft_DataDim - 2; i > 0; i--) {
    StockhamGenerator::hcKernWrite(transKernel, 3) << offset << " += (g_index/numGroupsY_" << i
                                << ")*" << stride[i + 1] << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "g_index = g_index % numGroupsY_" << i << ";"
                                << std::endl;
  }

  StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
}

// Small snippet of code that multiplies the twiddle factors into the
// butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the
// transpose
hcfftStatus genTwiddleMath(const hcfftPlanHandle plHandle,
                           const FFTKernelGenKeyParams& params,
                           std::stringstream& transKernel,
                           const std::string& dtComplex, bool fwd) {
  StockhamGenerator::hcKernWrite(transKernel, 9) << std::endl;

  StockhamGenerator::hcKernWrite(transKernel, 9)
      << dtComplex << " Wm = TW3step" << plHandle
      << "( (t_gx_p*32 + lidx) * (t_gy_p*32 + lidy + loop*8)" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
  StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9)
      << dtComplex << " Wt = TW3step" << plHandle
      << "( (t_gy_p*32 + lidx) * (t_gx_p*32 + lidy + loop*8)" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
  StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;

  StockhamGenerator::hcKernWrite(transKernel, 9) << dtComplex << " Tm, Tt;" << std::endl;

  if (fwd) {
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.x = ( Wm.x * tmpm.x ) - ( Wm.y * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.y = ( Wm.y * tmpm.x ) + ( Wm.x * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.x = ( Wt.x * tmpt.x ) - ( Wt.y * tmpt.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.y = ( Wt.y * tmpt.x ) + ( Wt.x * tmpt.y );" << std::endl;
  } else {
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.x =  ( Wm.x * tmpm.x ) + ( Wm.y * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.y = -( Wm.y * tmpm.x ) + ( Wm.x * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.x =  ( Wt.x * tmpt.x ) + ( Wt.y * tmpt.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.y = -( Wt.y * tmpt.x ) + ( Wt.x * tmpt.y );" << std::endl;
  }

  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpm.x = Tm.x;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpm.y = Tm.y;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpt.x = Tt.x;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpt.y = Tt.y;" << std::endl;

  StockhamGenerator::hcKernWrite(transKernel, 9) << std::endl;

  return HCFFT_SUCCEEDS;
}

// Small snippet of code that multiplies the twiddle factors into the
// butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the
// transpose
hcfftStatus genTwiddleMathLeadingDimensionBatched(
    const hcfftPlanHandle plHandle, const FFTKernelGenKeyParams& params,
    std::stringstream& transKernel, const std::string& dtComplex, bool fwd) {

  StockhamGenerator::hcKernWrite(transKernel, 9) << std::endl;
  if (params.fft_N[0] > params.fft_N[1]) {
    StockhamGenerator::hcKernWrite(transKernel, 9) << dtComplex << " Wm = TW3step" << plHandle
                                << " ( (" << params.fft_N[1]
                                << " * square_matrix_index + t_gx_p*32 + lidx) "
                                   "* (t_gy_p*32 + lidy + loop*8) "
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
    StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << dtComplex << " Wt = TW3step" << plHandle
                                << " ( (" << params.fft_N[1]
                                << " * square_matrix_index + t_gy_p*32 + lidx) "
                                   "* (t_gx_p*32 + lidy + loop*8) "
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
    StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
  } else {
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << dtComplex << " Wm = TW3step" << plHandle
        << " ( (t_gx_p*32 + lidx) * (" << params.fft_N[0]
        << " * square_matrix_index + t_gy_p*32 + lidy + loop*8) " << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
    StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << dtComplex << " Wt = TW3step" << plHandle
        << " ( (t_gy_p*32 + lidx) * (" << params.fft_N[0]
        << " * square_matrix_index + t_gx_p*32 + lidy + loop*8) " << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
    StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
  }
  StockhamGenerator::hcKernWrite(transKernel, 9) << dtComplex << " Tm, Tt;" << std::endl;

  if (fwd) {
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.x = ( Wm.x * tmpm.x ) - ( Wm.y * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.y = ( Wm.y * tmpm.x ) + ( Wm.x * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.x = ( Wt.x * tmpt.x ) - ( Wt.y * tmpt.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.y = ( Wt.y * tmpt.x ) + ( Wt.x * tmpt.y );" << std::endl;
  } else {
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.x =  ( Wm.x * tmpm.x ) + ( Wm.y * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tm.y = -( Wm.y * tmpm.x ) + ( Wm.x * tmpm.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.x =  ( Wt.x * tmpt.x ) + ( Wt.y * tmpt.y );" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 9)
        << "Tt.y = -( Wt.y * tmpt.x ) + ( Wt.x * tmpt.y );" << std::endl;
  }

  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpm.x = Tm.x;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpm.y = Tm.y;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpt.x = Tt.x;" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 9) << "tmpt.y = Tt.y;" << std::endl;

  StockhamGenerator::hcKernWrite(transKernel, 9) << std::endl;

  return HCFFT_SUCCEEDS;
}

hcfftStatus genTransposePrototype(
    const FFTKernelGenKeyParams& params, const size_t& lwSize,
    const std::string& dtPlanar, const std::string& dtComplex,
    const std::string& funcName, std::stringstream& transKernel,
    std::string& dtInput, std::string& dtOutput, bool twiddleTransposeKernel) {
  uint arg = 0;
  // Declare and define the function
  StockhamGenerator::hcKernWrite(transKernel, 0) << "extern \"C\"\n { void" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 0)
      << funcName << "(  std::map<int, void*> vectArr, uint batchSize, "
                     "accelerator_view &acc_view, accelerator &acc) \n {";

  switch (params.fft_inputLayout) {
    case HCFFT_COMPLEX_INTERLEAVED:
      dtInput = dtComplex;
      dtOutput = dtComplex;
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      break;
    case HCFFT_COMPLEX_PLANAR:
      dtInput = dtPlanar;
      dtOutput = dtPlanar;
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA_R"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA_I"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      break;
    case HCFFT_HERMITIAN_INTERLEAVED:
    case HCFFT_HERMITIAN_PLANAR:
      return HCFFT_INVALID;
    case HCFFT_REAL:
      dtInput = dtPlanar;
      dtOutput = dtPlanar;
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      break;
    default:
      return HCFFT_INVALID;
  }

  if (params.fft_placeness == HCFFT_OUTOFPLACE) {
    switch (params.fft_outputLayout) {
      case HCFFT_COMPLEX_INTERLEAVED:
        dtInput = dtComplex;
        dtOutput = dtComplex;
        StockhamGenerator::hcKernWrite(transKernel, 0) << dtOutput << " * outputA"
                                    << " = static_cast< " << dtOutput
                                    << "*> (vectArr[" << arg++ << "]);";
        break;
      case HCFFT_COMPLEX_PLANAR:
        dtInput = dtPlanar;
        dtOutput = dtPlanar;
        StockhamGenerator::hcKernWrite(transKernel, 0) << dtOutput << " * outputA_R"
                                    << " = static_cast< " << dtOutput
                                    << "*> (vectArr[" << arg++ << "]);";
        StockhamGenerator::hcKernWrite(transKernel, 0) << dtOutput << " * outputA_I"
                                    << " = static_cast< " << dtOutput
                                    << "*> (vectArr[" << arg++ << "]);";
        break;
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      case HCFFT_REAL:
        dtInput = dtPlanar;
        dtOutput = dtPlanar;
        StockhamGenerator::hcKernWrite(transKernel, 0) << dtOutput << " * outputA"
                                    << " = static_cast< " << dtOutput
                                    << "*> (vectArr[" << arg++ << "]);";
        break;
      default:
        return HCFFT_INVALID;
    }
  }

  if (twiddleTransposeKernel) {
    StockhamGenerator::hcKernWrite(transKernel, 0) << dtComplex << " *" << StockhamGenerator::TwTableLargeName()
                                << " = static_cast< " << dtComplex
                                << "*> (vectArr[" << arg++ << "]);";
  }
  return HCFFT_SUCCEEDS;
}

hcfftStatus genTransposePrototypeLeadingDimensionBatched(
    const FFTKernelGenKeyParams& params, const size_t& lwSize,
    const std::string& dtPlanar, const std::string& dtComplex,
    const std::string& funcName, std::stringstream& transKernel,
    std::string& dtInput, std::string& dtOutput, bool genTwiddle) {

  uint arg = 0;
  // Declare and define the function
  StockhamGenerator::hcKernWrite(transKernel, 0) << "extern \"C\"\n { void" << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 0)
      << funcName << "(  std::map<int, void*> vectArr, uint batchSize, "
                     "accelerator_view &acc_view, accelerator &acc) \n {";

  switch (params.fft_inputLayout) {
    case HCFFT_COMPLEX_INTERLEAVED:
      dtInput = dtComplex;
      dtOutput = dtComplex;
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      break;
    case HCFFT_COMPLEX_PLANAR:
      dtInput = dtPlanar;
      dtOutput = dtPlanar;
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA_R"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA_I"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      break;
    case HCFFT_HERMITIAN_INTERLEAVED:
    case HCFFT_HERMITIAN_PLANAR:
      return HCFFT_INVALID;
    case HCFFT_REAL:
      dtInput = dtPlanar;
      dtOutput = dtPlanar;
      StockhamGenerator::hcKernWrite(transKernel, 0) << dtInput << " * inputA"
                                  << " = static_cast< " << dtInput
                                  << "*> (vectArr[" << arg++ << "]);";
      break;
    default:
      return HCFFT_INVALID;
  }

  if (genTwiddle) {
    StockhamGenerator::hcKernWrite(transKernel, 0) << dtComplex << " *" << StockhamGenerator::TwTableLargeName()
                                << " = static_cast< " << dtComplex
                                << "*> (vectArr[" << arg++ << "]);";
  }

  return HCFFT_SUCCEEDS;
}

/* -> get_cycles function gets the swapping logic required for given row x col
matrix.
-> cycle_map[0] holds the total number of cycles required.
-> cycles start and end with the same index, hence we can identify individual
cycles,
though we tend to store the cycle index contiguously*/
void get_cycles(size_t* cycle_map, size_t num_reduced_row,
                size_t num_reduced_col) {
  int* is_swapped = new int[num_reduced_row * num_reduced_col];
  int i, map_index = 1, num_cycles = 0;
  size_t swap_id;
  /*initialize swap map*/
  is_swapped[0] = 1;
  is_swapped[num_reduced_row * num_reduced_col - 1] = 1;
  for (i = 1; i < (num_reduced_row * num_reduced_col - 1); i++) {
    is_swapped[i] = 0;
  }

  for (i = 1; i < (num_reduced_row * num_reduced_col - 1); i++) {
    swap_id = i;
    while (!is_swapped[swap_id]) {
      is_swapped[swap_id] = 1;
      cycle_map[map_index++] = swap_id;
      swap_id =
          (num_reduced_row * swap_id) % (num_reduced_row * num_reduced_col - 1);
      if (swap_id == i) {
        cycle_map[map_index++] = swap_id;
        num_cycles++;
      }
    }
  }
  cycle_map[0] = num_cycles;
  delete[] is_swapped;
}

/*
calculate the permutation cycles consumed in swap kernels.
each cycle is strored in a vecotor. hopfully there are mutliple independent
vectors thus we use a vector of vecotor
*/
void permutation_calculation(
    size_t m, size_t n, std::vector<std::vector<size_t> >& permutationVec) {
  /*
  calculate inplace transpose permutation lists
  reference:
  https://en.wikipedia.org/wiki/In-place_matrix_transposition
  and
  http://www.netlib.org/utk/people/JackDongarra/CCDSC-2014/talk35.pdf
  row major matrix of size n x m
  p(k) = (k*n)mod(m*n-1), if 0 < k < m*n-1
  when k = 0 or m*n-1, it does not require movement
  */
  if (m < 1 || n < 1) return;

  size_t mn_minus_one = m * n - 1;
  // maintain a table so check is faster
  size_t* table = new size_t[mn_minus_one + 1]();  // init to zeros
  table[0] = 1;

  for (size_t i = 1; i < mn_minus_one; i++) {
    // first check if i is already stored in somewhere in vector of vectors
    bool already_checked = false;
    if (table[i] >= 1) already_checked = true;
    if (already_checked == true) continue;

    // if not checked yet
    std::vector<size_t> vec;
    vec.push_back(i);
    table[i] += 1;
    size_t temp = i;

    while (1) {
      temp = (temp * n);
      temp = temp % (mn_minus_one);
      if (find(vec.begin(), vec.end(), temp) != vec.end()) {
        // what goes around comes around and it should
        break;
      }
      if (table[temp] >= 1) {
        already_checked = true;
        break;
      }
      vec.push_back(temp);
      table[temp] += 1;
    }
    if (already_checked == true) continue;
    permutationVec.push_back(vec);
  }
  delete[] table;
}
// swap lines. This kind of kernels are using with combination of square
// transpose kernels to perform nonsqaure transpose
// this function assumes a 1:2 ratio
hcfftStatus genSwapKernel(const FFTKernelGenKeyParams& params,
                          std::string& strKernel, std::string& KernelFuncName,
                          const size_t& lwSize, const size_t reShapeFactor,
                          std::vector<size_t> gWorkSize,
                          std::vector<size_t> lWorkSize, size_t count) {
  strKernel.reserve(4096);
  std::stringstream transKernel(std::stringstream::out);

  // These strings represent the various data types we read or write in the
  // kernel, depending on how the plan
  // is configured
  std::string dtInput;   // The type read as input into kernel
  std::string dtOutput;  // The type written as output from kernel
  std::string dtPlanar;  // Fundamental type for planar arrays
  std::string tmpBuffType;
  std::string dtComplex;  // Fundamental type for complex arrays

  switch (params.fft_precision) {
    case HCFFT_SINGLE:
      dtPlanar = "float";
      dtComplex = "float_2";
      break;
    case HCFFT_DOUBLE:
      dtPlanar = "double";
      dtComplex = "double2";
      break;
    default:
      return HCFFT_INVALID;
      break;
  }

  // This detects whether the input matrix is rectangle of ratio 1:2

  if ((params.fft_N[0] != 2 * params.fft_N[1]) &&
      (params.fft_N[1] != 2 * params.fft_N[0])) {
    return HCFFT_INVALID;
  }

  if (params.fft_placeness == HCFFT_OUTOFPLACE) {
    return HCFFT_INVALID;
  }

  size_t smaller_dim =
      (params.fft_N[0] < params.fft_N[1]) ? params.fft_N[0] : params.fft_N[1];

  size_t input_elm_size_in_bytes;
  switch (params.fft_precision) {
    case HCFFT_SINGLE:
      input_elm_size_in_bytes = 4;
      break;
    case HCFFT_DOUBLE:
      input_elm_size_in_bytes = 8;
      break;
    default:
      return HCFFT_INVALID;
  }

  switch (params.fft_outputLayout) {
    case HCFFT_COMPLEX_INTERLEAVED:
    case HCFFT_COMPLEX_PLANAR:
      input_elm_size_in_bytes *= 2;
      break;
    case HCFFT_REAL:
      break;
    default:
      return HCFFT_INVALID;
  }
  size_t max_elements_loaded = AVAIL_MEM_SIZE / input_elm_size_in_bytes;
  size_t num_elements_loaded;
  size_t local_work_size_swap, num_grps_pro_row;

  tmpBuffType = " tile_static ";
  if ((max_elements_loaded >> 1) > smaller_dim) {
    local_work_size_swap = (smaller_dim < 256) ? smaller_dim : 256;
    num_elements_loaded = smaller_dim;
    num_grps_pro_row = 1;
  } else {
    num_grps_pro_row = (smaller_dim << 1) / max_elements_loaded;
    num_elements_loaded = max_elements_loaded >> 1;
    local_work_size_swap =
        (num_elements_loaded < 256) ? num_elements_loaded : 256;
  }

  /*Generating the  swapping logic*/
  {
    size_t num_reduced_row;
    size_t num_reduced_col;

    if (params.fft_N[1] == smaller_dim) {
      num_reduced_row = smaller_dim;
      num_reduced_col = 2;
    } else {
      num_reduced_row = 2;
      num_reduced_col = smaller_dim;
    }

    std::string funcName;

    StockhamGenerator::hcKernWrite(transKernel, 0) << std::endl;

    size_t* cycle_map = new size_t[num_reduced_row * num_reduced_col * 2];
    /* The memory required by cycle_map cannot exceed 2 times row*col by
     * design*/

    get_cycles(cycle_map, num_reduced_row, num_reduced_col);

    size_t *cycle_stat = new size_t[cycle_map[0] * 2], stat_idx = 0;
    StockhamGenerator::hcKernWrite(transKernel, 0) << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 0) << "size_t swap_table[][3] = {" << std::endl;

    size_t inx = 0, start_inx, swap_inx = 0, num_swaps = 0;
    for (size_t i = 0; i < cycle_map[0]; i++) {
      start_inx = cycle_map[++inx];
      StockhamGenerator::hcKernWrite(transKernel, 0) << "{  " << start_inx << ",  "
                                  << cycle_map[inx + 1] << ",  0},"
                                  << std::endl;
      cycle_stat[stat_idx++] = num_swaps;
      num_swaps++;

      while (start_inx != cycle_map[++inx]) {
        size_t action_var = (cycle_map[inx + 1] == start_inx) ? 2 : 1;
        StockhamGenerator::hcKernWrite(transKernel, 0) << "{  " << cycle_map[inx] << ",  "
                                    << cycle_map[inx + 1] << ",  " << action_var
                                    << "}," << std::endl;
        if (action_var == 2) cycle_stat[stat_idx++] = num_swaps;
        num_swaps++;
      }
    }
    /*Appending swap table for touching corner elements for post call back*/
    size_t last_datablk_idx = num_reduced_row * num_reduced_col - 1;
    StockhamGenerator::hcKernWrite(transKernel, 0) << "{  0,  0,  0}," << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 0) << "{  " << last_datablk_idx << ",  "
                                << last_datablk_idx << ",  0}," << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 0) << "};" << std::endl;
    /*cycle_map[0] + 2, + 2 is added for post callback table appending*/
    size_t num_cycles_minus_1 = cycle_map[0] - 1;

    StockhamGenerator::hcKernWrite(transKernel, 0) << "size_t cycle_stat[" << cycle_map[0]
                                << "][2] = {" << std::endl;
    for (size_t i = 0; i < num_cycles_minus_1; i++) {
      StockhamGenerator::hcKernWrite(transKernel, 0) << "{  " << cycle_stat[i * 2] << ",  "
                                  << cycle_stat[i * 2 + 1] << "}," << std::endl;
    }
    StockhamGenerator::hcKernWrite(transKernel, 0)
        << "{  " << cycle_stat[num_cycles_minus_1 * 2] << ",  "
        << (cycle_stat[num_cycles_minus_1 * 2 + 1] + 2) << "}," << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 0) << "};" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 0) << std::endl;

    switch (params.fft_inputLayout) {
      case HCFFT_COMPLEX_INTERLEAVED:
        StockhamGenerator::hcKernWrite(transKernel, 0)
            << "void swap( " << dtComplex << "* inputA, " << tmpBuffType << " "
            << dtComplex << "* Ls, " << tmpBuffType << " " << dtComplex
            << " * Ld, size_t is, size_t id, size_t pos, size_t end_indx, "
               "size_t work_id, size_t inOffset";
        break;
      case HCFFT_COMPLEX_PLANAR:
        StockhamGenerator::hcKernWrite(transKernel, 0)
            << "void swap( " << dtPlanar << "* inputA_R, " << dtPlanar
            << "* inputA_I, " << tmpBuffType << " " << dtComplex << "* Ls, "
            << tmpBuffType << " " << dtComplex
            << "* Ld, size_t is, size_t id, size_t pos, size_t end_indx, "
               "size_t work_id, size_t inOffset";
        break;
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      case HCFFT_REAL:
        StockhamGenerator::hcKernWrite(transKernel, 0)
            << "void swap( " << dtPlanar << "* inputA, " << tmpBuffType << " "
            << dtPlanar << "* Ls, " << tmpBuffType << " " << dtPlanar
            << "* Ld, size_t is, size_t id, size_t pos, size_t end_indx, "
               "size_t work_id, size_t inOffset";
        break;
      default:
        return HCFFT_INVALID;
    }

    StockhamGenerator::hcKernWrite(transKernel, 0) << "){" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "for (size_t j = tidx.local[0]; j < end_indx; j += "
        << local_work_size_swap << "){" << std::endl;

    switch (params.fft_inputLayout) {
      case HCFFT_REAL:
      case HCFFT_COMPLEX_INTERLEAVED:
        StockhamGenerator::hcKernWrite(transKernel, 6) << "if (pos == 0){" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ls[j] = inputA[inOffset + is *" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ld[j] = inputA[inOffset + id *" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 6) << "else if (pos == 1){" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ld[j] = inputA[inOffset + id *" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 6) << "inputA[inOffset + id*" << smaller_dim
                                    << " + " << num_elements_loaded
                                    << " * work_id + j] = Ls[j];" << std::endl;
        break;
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      case HCFFT_COMPLEX_PLANAR:
        StockhamGenerator::hcKernWrite(transKernel, 6) << "if (pos == 0){" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ls[j].x = inputA_R[inOffset + is*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ls[j].y = inputA_I[inOffset + is*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ld[j].x = inputA_R[inOffset + id*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ld[j].y = inputA_I[inOffset + id*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 6) << "else if (pos == 1){" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ld[j].x = inputA_R[inOffset + id*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 9)
            << "Ld[j].y = inputA_I[inOffset + id*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 6)
            << "inputA_R[inOffset + id*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j] = Ls[j].x;" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 6)
            << "inputA_I[inOffset + id*" << smaller_dim << " + "
            << num_elements_loaded << " * work_id + j] = Ls[j].y;" << std::endl;
        break;
      default:
        return HCFFT_INVALID;
    }
    StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 0) << "}" << std::endl << std::endl;

    funcName = "swap_nonsquare";
    funcName += SztToStr(count);

    KernelFuncName = funcName;
    // Generate kernel API

    /*when swap can be performed in LDS itself then, same prototype of transpose
     * can be used for swap function too*/
    genTransposePrototypeLeadingDimensionBatched(
        params, local_work_size_swap, dtPlanar, dtComplex, funcName,
        transKernel, dtInput, dtOutput, false);
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::extent<2> grdExt( ";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(gWorkSize[0]) << ", 1); \n"
                                << "\thc::tiled_extent<2> t_ext = grdExt.tile(";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(lwSize) << ", 1);\n";
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::parallel_for_each(acc_view, t_ext, "
                                   "[=] (hc::tiled_index<2> tidx) [[hc]]\n\t "
                                   "{ ";

    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t g_index = tidx.tile[0];"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "const size_t numGroupsY_1 = " << cycle_map[0] * num_grps_pro_row
        << " ;" << std::endl;
    for (size_t i = 2; i < params.fft_DataDim - 1; i++) {
      StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t numGroupsY_" << i
                                  << " = numGroupsY_" << i - 1 << " * "
                                  << params.fft_N[i] << ";" << std::endl;
    }

    delete[] cycle_map;
    delete[] cycle_stat;

    Swap_OffsetCalc(transKernel, params);

    // Handle planar and interleaved right here
    switch (params.fft_inputLayout) {
      case HCFFT_COMPLEX_INTERLEAVED:
      case HCFFT_REAL:

        StockhamGenerator::hcKernWrite(transKernel, 3)
            << "tile_static " << dtInput << " tmp_tot_mem["
            << (num_elements_loaded * 2) << "];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 3) << tmpBuffType << " " << dtInput
                                    << " *te = tmp_tot_mem;" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 3) << tmpBuffType << " " << dtInput
                                    << " *to = (tmp_tot_mem + "
                                    << num_elements_loaded << ");" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 3)
            << "uint inOffset = iOffset;"
            << std::endl;  // Set A ptr to the start of each slice
        break;
      case HCFFT_COMPLEX_PLANAR:

        StockhamGenerator::hcKernWrite(transKernel, 3)
            << "tile_static " << dtComplex << " tmp_tot_mem["
            << (num_elements_loaded * 2) << "];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 3) << tmpBuffType << " " << dtComplex
                                    << " *te = tmp_tot_mem;" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 3) << tmpBuffType << " " << dtComplex
                                    << " *to = (tmp_tot_mem + "
                                    << num_elements_loaded << ");" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 3)
            << "uint inOffset = iOffset;"
            << std::endl;  // Set A ptr to the start of each slice
        break;
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;

      default:
        return HCFFT_INVALID;
    }

    switch (params.fft_inputLayout) {
      case HCFFT_COMPLEX_INTERLEAVED:
      case HCFFT_COMPLEX_PLANAR:
        StockhamGenerator::hcKernWrite(transKernel, 3) << tmpBuffType << " " << dtComplex
                                    << " *tmp_swap_ptr[2];" << std::endl;
        break;
      case HCFFT_REAL:
        StockhamGenerator::hcKernWrite(transKernel, 3) << tmpBuffType << " " << dtPlanar
                                    << " *tmp_swap_ptr[2];" << std::endl;
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        break;
    }
    StockhamGenerator::hcKernWrite(transKernel, 3) << "tmp_swap_ptr[0] = te;" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "tmp_swap_ptr[1] = to;" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t swap_inx = 0;" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t start = cycle_stat[g_index / "
                                << num_grps_pro_row << "][0];" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t end = cycle_stat[g_index / "
                                << num_grps_pro_row << "][1];" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t end_indx = " << num_elements_loaded
                                << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t work_id = g_index % "
                                << num_grps_pro_row << ";" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "if( work_id == " << (num_grps_pro_row - 1)
                                << " ){" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 6)
        << "end_indx = "
        << smaller_dim - num_elements_loaded * (num_grps_pro_row - 1) << ";"
        << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "for (size_t loop = start; loop <= end; loop ++){" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 6) << "swap_inx = 1 - swap_inx;" << std::endl;

    switch (params.fft_inputLayout) {
      case HCFFT_COMPLEX_INTERLEAVED:
      case HCFFT_REAL:
        StockhamGenerator::hcKernWrite(transKernel, 6)
            << "swap(inputA, tmp_swap_ptr[swap_inx], tmp_swap_ptr[1 - "
               "swap_inx], swap_table[loop][0], swap_table[loop][1], "
               "swap_table[loop][2], end_indx, work_id, inOffset";
        break;
      case HCFFT_COMPLEX_PLANAR:
        StockhamGenerator::hcKernWrite(transKernel, 6)
            << "swap(inputA_R, inputA_I, tmp_swap_ptr[swap_inx], "
               "tmp_swap_ptr[1 - swap_inx], swap_table[loop][0], "
               "swap_table[loop][1], swap_table[loop][2], end_indx, work_id, "
               "inOffset";
        break;
      case HCFFT_HERMITIAN_PLANAR:
      case HCFFT_HERMITIAN_INTERLEAVED:
        break;
    }
    StockhamGenerator::hcKernWrite(transKernel, 0) << ");" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 0) << "}).wait();\n}}\n" << std::endl;
    strKernel = transKernel.str();
  }
  return HCFFT_SUCCEEDS;
}

// swap lines. a more general kernel generator.
// this function accepts any ratio in theory. But in practice we restrict it to
// 1:2, 1:3, 1:5 and 1:10 ration
hcfftStatus genSwapKernelGeneral(
    void** twiddleslarge, hc::accelerator acc, const hcfftPlanHandle plHandle,
    const FFTKernelGenKeyParams& params, std::string& strKernel,
    std::string& KernelFuncName, const size_t& lwSize,
    const size_t reShapeFactor, std::vector<size_t> gWorkSize,
    std::vector<size_t> lWorkSize, size_t count) {
  if (params.fft_placeness == HCFFT_OUTOFPLACE) return HCFFT_INVALID;

  size_t smaller_dim =
      (params.fft_N[0] < params.fft_N[1]) ? params.fft_N[0] : params.fft_N[1];
  size_t bigger_dim =
      (params.fft_N[0] >= params.fft_N[1]) ? params.fft_N[0] : params.fft_N[1];
  size_t dim_ratio = bigger_dim / smaller_dim;
  /*
  if ( (params.fft_N[0] != 2 * params.fft_N[1]) && (params.fft_N[1] != 2 *
  params.fft_N[0]) &&
           (params.fft_N[0] != 3 * params.fft_N[1]) && (params.fft_N[1] != 3 *
  params.fft_N[0]) &&
           (params.fft_N[0] != 5 * params.fft_N[1]) && (params.fft_N[1] != 5 *
  params.fft_N[0]) &&
           (params.fft_N[0] != 10 * params.fft_N[1]) && (params.fft_N[1] != 10 *
  params.fft_N[0]) )
*/
  if (dim_ratio % 2 != 0 && dim_ratio % 3 != 0 && dim_ratio % 5 != 0 &&
      dim_ratio % 10 != 0) {
    return HCFFT_INVALID;
  }

  strKernel.reserve(4096);
  std::stringstream transKernel(std::stringstream::out);

  // These strings represent the various data types we read or write in the
  // kernel, depending on how the plan
  // is configured
  std::string dtInput;   // The type read as input into kernel
  std::string dtOutput;  // The type written as output from kernel
  std::string dtPlanar;  // Fundamental type for planar arrays
  std::string tmpBuffType;
  std::string dtComplex;  // Fundamental type for complex arrays

  switch (params.fft_precision) {
    case HCFFT_SINGLE:
      dtPlanar = "float";
      dtComplex = "float_2";
      break;
    case HCFFT_DOUBLE:
      dtPlanar = "double";
      dtComplex = "double2";
      break;
    default:
      return HCFFT_INVALID;
      break;
  }

  size_t LDS_per_WG = smaller_dim;
  while (LDS_per_WG > 1024) {  // avoiding using too much lds memory.
                              //  the biggest
                             // LDS memory we will allocate would be
                             // 1024*sizeof(float_2/double2)*2
    if (LDS_per_WG % 2 == 0) {
      LDS_per_WG /= 2;
      continue;
    }
    if (LDS_per_WG % 3 == 0) {
      LDS_per_WG /= 3;
      continue;
    }
    if (LDS_per_WG % 5 == 0) {
      LDS_per_WG /= 5;
      continue;
    }
    return HCFFT_INVALID;
  }
  size_t WG_per_line = smaller_dim / LDS_per_WG;

  size_t input_elm_size_in_bytes;
  switch (params.fft_precision) {
    case HCFFT_SINGLE:
      input_elm_size_in_bytes = 4;
      break;
    case HCFFT_DOUBLE:
      input_elm_size_in_bytes = 8;
      break;
    default:
      return HCFFT_INVALID;
  }

  switch (params.fft_outputLayout) {
    case HCFFT_COMPLEX_INTERLEAVED:
    case HCFFT_COMPLEX_PLANAR:
      input_elm_size_in_bytes *= 2;
      break;
    case HCFFT_REAL:
      break;
    default:
      return HCFFT_INVALID;
  }
  /* not entirely clearly why do i need this yet
  size_t max_elements_loaded = AVAIL_MEM_SIZE / input_elm_size_in_bytes;
  size_t num_elements_loaded;
  size_t local_work_size_swap, num_grps_pro_row;
  */

  // twiddle in swap kernel (for now, swap with twiddle seems to always be the
  // second kernel after transpose)
  bool twiddleSwapKernel = params.fft_3StepTwiddle && (dim_ratio > 1);
  // twiddle factors applied to the output of swap kernels if swap kernels are
  // the last kernel in transpose order
  bool twiddleSwapKernelOut =
      twiddleSwapKernel &&
      (params.nonSquareKernelOrder == TRANSPOSE_AND_SWAP ||
       params.nonSquareKernelOrder == TRANSPOSE_LEADING_AND_SWAP);
  // twiddle factors applied to the input of swap kernels if swap kernels are
  // the first kernel in transpose order
  bool twiddleSwapKernelIn =
      twiddleSwapKernel && (params.nonSquareKernelOrder == SWAP_AND_TRANSPOSE);

  // generate the swap_table
  std::vector<std::vector<size_t> > permutationTable;
  permutation_calculation(dim_ratio, smaller_dim, permutationTable);

  StockhamGenerator::hcKernWrite(transKernel, 0) << "size_t swap_table["
                              << permutationTable.size() + 2 << "][1] = {"
                              << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 0) << "{0}," << std::endl;
  StockhamGenerator::hcKernWrite(transKernel, 0) << "{" << smaller_dim * dim_ratio - 1 << "},"
                              << std::endl;  // add the first and last row to
                                             // the swap table. needed for
                                             // twiddling
  for (std::vector<std::vector<size_t> >::iterator itor =
           permutationTable.begin();
       itor != permutationTable.end(); itor++) {
    StockhamGenerator::hcKernWrite(transKernel, 0) << "{" << (*itor)[0] << "}";
    if (itor == (permutationTable.end() - 1))  // last vector
      StockhamGenerator::hcKernWrite(transKernel, 0) << std::endl << "};" << std::endl;
    else
      StockhamGenerator::hcKernWrite(transKernel, 0) << "," << std::endl;
  }

  // twiddle in swap kernel
  // twiddle in or out should be using the same twiddling table
  if (twiddleSwapKernel) {
    std::string str;
    if (params.fft_precision == HCFFT_SINGLE) {
      StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2,
                                           StockhamGenerator::P_SINGLE>
          twLarge(smaller_dim * smaller_dim * dim_ratio);
      twLarge.GenerateTwiddleTable(str, plHandle);
      twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
    } else {
      StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2,
                                           StockhamGenerator::P_DOUBLE>
          twLarge(smaller_dim * smaller_dim * dim_ratio);
      twLarge.GenerateTwiddleTable(str, plHandle);
      twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
    }
    StockhamGenerator::hcKernWrite(transKernel, 0) << str << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 0) << std::endl;
  }

  // std::string funcName = "swap_nonsquare_" + std::to_string(smaller_dim) +
  // "_" + std::to_string(dim_ratio);
  std::string funcName = "swap_nonsquare_";
  std::string smaller_dim_str = SztToStr(smaller_dim);
  std::string dim_ratio_str = SztToStr(dim_ratio);
  if (params.fft_N[0] > params.fft_N[1])
    funcName = funcName + smaller_dim_str + "_" + dim_ratio_str;
  else
    funcName = funcName + dim_ratio_str + "_" + smaller_dim_str;

  funcName += SztToStr(count);
  KernelFuncName = funcName;
  size_t local_work_size_swap = 256;

  for (size_t bothDir = 0; bothDir < 2; bothDir++) {
    bool fwd = bothDir ? false : true;
    // Generate kernel API

    /*when swap can be performed in LDS itself then, same prototype of transpose
     * can be used for swap function too*/
    std::string funcNameTW;
    if (twiddleSwapKernel) {
      if (fwd)
        funcNameTW = funcName + "_tw_fwd";
      else
        funcNameTW = funcName + "_tw_back";
    } else {
      funcNameTW = funcName;
    }

    genTransposePrototypeLeadingDimensionBatched(
        params, local_work_size_swap, dtPlanar, dtComplex, funcNameTW,
        transKernel, dtInput, dtOutput, twiddleSwapKernel);
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::extent<2> grdExt( ";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(gWorkSize[0]) << ", 1); \n"
                                << "\thc::tiled_extent<2> t_ext = grdExt.tile(";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(lwSize) << ", 1);\n";
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::parallel_for_each(acc_view, t_ext, "
                                   "[=] (hc::tiled_index<2> tidx) [[hc]]\n\t "
                                   "{ ";

    StockhamGenerator::hcKernWrite(transKernel, 3) << "//each wg handles 1/" << WG_per_line
                                << " row of " << LDS_per_WG << " in memory"
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "const size_t num_wg_per_batch = "
        << (permutationTable.size() + 2) * WG_per_line << ";"
        << std::endl;  // number of wg per batch = number of independent cycles
    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t group_id = tidx.tile[0];"
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t idx = tidx.local[0];" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "size_t batch_offset = group_id / num_wg_per_batch;" << std::endl;
    switch (params.fft_inputLayout) {
      case HCFFT_REAL:
      case HCFFT_COMPLEX_INTERLEAVED:
        StockhamGenerator::hcKernWrite(transKernel, 3) << " uint inOffset = batch_offset*"
                                    << smaller_dim * bigger_dim << ";"
                                    << std::endl;
        break;
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      case HCFFT_COMPLEX_PLANAR: {
        StockhamGenerator::hcKernWrite(transKernel, 3) << "uint inOffset = batch_offset*"
                                    << smaller_dim * bigger_dim << ";"
                                    << std::endl;
        break;
      }
      default:
        return HCFFT_INVALID;
    }
    StockhamGenerator::hcKernWrite(transKernel, 3) << "group_id -= batch_offset*"
                                << (permutationTable.size() + 2) * WG_per_line
                                << ";" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    if (WG_per_line == 1)
      StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t prev = swap_table[group_id][0];"
                                  << std::endl;
    else
      StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t prev = swap_table[group_id/"
                                  << WG_per_line << "][0];" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t next = 0;" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    switch (params.fft_inputLayout) {
      case HCFFT_REAL:
      case HCFFT_COMPLEX_INTERLEAVED: {
        StockhamGenerator::hcKernWrite(transKernel, 3)
            << "tile_static " << dtInput << " prevValue[" << LDS_per_WG << "];"
            << std::endl;  // lds within each wg should be able to store a row
                           // block (smaller_dim) of element
        StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtInput
                                    << " nextValue[" << LDS_per_WG << "];"
                                    << std::endl;
        break;
      }
      case HCFFT_COMPLEX_PLANAR: {
        StockhamGenerator::hcKernWrite(transKernel, 3)
            << "tile_static " << dtComplex << " prevValue[" << LDS_per_WG
            << "];" << std::endl;  // lds within each wg should be able to store
                                   // a row block (smaller_dim) of element
        StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtComplex
                                    << " nextValue[" << LDS_per_WG << "];"
                                    << std::endl;
        break;
      }
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      default:
        return HCFFT_INVALID;
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    if (params.fft_N[0] >
        params.fft_N[1]) {  // decides whether we have a tall or wide rectangle
      if (WG_per_line == 1) {
        // might look like: size_t group_offset = (prev/3)*729*3 + (prev%3)*729;
        StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t group_offset = (prev/"
                                    << dim_ratio << ")*" << smaller_dim << "*"
                                    << dim_ratio << " + (prev%" << dim_ratio
                                    << ")*" << smaller_dim << ";" << std::endl;
      } else {
        // if smaller_dim is 2187 > 1024 this should look like size_t
        // group_offset = (prev/3)*2187*3 + (prev%3)*2187 + (group_id % 3)*729;
        StockhamGenerator::hcKernWrite(transKernel, 3)
            << "size_t group_offset = (prev/" << dim_ratio << ")*"
            << smaller_dim << "*" << dim_ratio << " + (prev%" << dim_ratio
            << ")*" << smaller_dim << " + (group_id % " << WG_per_line << ")*"
            << LDS_per_WG << ";" << std::endl;
      }
    } else {
      if (WG_per_line == 1)  // might look like: size_t group_offset = prev*729;
        StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t group_offset = (prev*"
                                    << smaller_dim << ");" << std::endl;
      else  // if smaller_dim is 2187 > 1024 this should look like size_t
            // group_offset = prev*2187 + (group_id % 3)*729;
        StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t group_offset = (prev*"
                                    << smaller_dim << ") + (group_id % "
                                    << WG_per_line << ")*" << LDS_per_WG << ";"
                                    << std::endl;
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    // move to that row block and load that row block to LDS
    if (twiddleSwapKernelIn) {
      StockhamGenerator::hcKernWrite(transKernel, 6) << "size_t p;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "size_t q;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << dtComplex << " twiddle_factor;"
                                  << std::endl;
    }
    switch (params.fft_inputLayout) {
      case HCFFT_REAL:
      case HCFFT_COMPLEX_INTERLEAVED: {
        for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
          if (i + 256 < LDS_per_WG) {
            if (twiddleSwapKernelIn) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.y + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].y * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].x * twiddle_factor.y;" << std::endl;
              }
            } else {
              StockhamGenerator::hcKernWrite(transKernel, 3)
                  << "prevValue[idx+" << i
                  << "] = inputA[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            }
          } else {
            // need to handle boundary
            StockhamGenerator::hcKernWrite(transKernel, 3) << "if(idx+" << i << "<" << LDS_per_WG
                                        << "){" << std::endl;
            if (twiddleSwapKernelIn) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.y + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].y * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].x * twiddle_factor.y;" << std::endl;
              }
            } else {
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "prevValue[idx+" << i
                  << "] = inputA[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            }
            StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;
          }
        }
        break;
      }
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      case HCFFT_COMPLEX_PLANAR: {
        for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
          if (i + 256 < LDS_per_WG) {
            if (twiddleSwapKernelIn) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.y + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA_I[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_R[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
              }
            } else {
              StockhamGenerator::hcKernWrite(transKernel, 3)
                  << "prevValue[idx+" << i
                  << "].x = inputA_R[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 3)
                  << "prevValue[idx+" << i
                  << "].y = inputA_I[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            }
          } else {
            // need to handle boundary
            StockhamGenerator::hcKernWrite(transKernel, 3) << "if(idx+" << i << "<" << LDS_per_WG
                                        << "){" << std::endl;
            if (twiddleSwapKernelIn) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.y + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "prevValue[idx+" << i
                    << "].y = inputA_I[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_R[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
              }
            } else {
              StockhamGenerator::hcKernWrite(transKernel, 3)
                  << "prevValue[idx+" << i
                  << "].x = inputA_R[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 3)
                  << "prevValue[idx+" << i
                  << "].y = inputA_I[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            }
            StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;
          }
        }
        break;
      }
      default:
        return HCFFT_INVALID;
    }
    StockhamGenerator::hcKernWrite(transKernel, 3) << "tidx.barrier.wait();" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "do{" << std::endl;  // begining of do-while
    // calculate the next location p(k) = (k*n)mod(m*n-1), if 0 < k < m*n-1
    if (params.fft_N[0] >
        params.fft_N[1]) {  // decides whether we have a tall or wide rectangle
      StockhamGenerator::hcKernWrite(transKernel, 6) << "next = (prev*" << smaller_dim << ")%"
                                  << smaller_dim * dim_ratio - 1 << ";"
                                  << std::endl;
      // takes care the last row
      StockhamGenerator::hcKernWrite(transKernel, 6)
          << "if (prev == " << smaller_dim * dim_ratio - 1 << ")" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "next = " << smaller_dim * dim_ratio - 1
                                  << ";" << std::endl;
      if (WG_per_line == 1) {
        StockhamGenerator::hcKernWrite(transKernel, 6)
            << "group_offset = (next/" << dim_ratio << ")*" << smaller_dim
            << "*" << dim_ratio << " + (next%" << dim_ratio << ")*"
            << smaller_dim << ";"
            << std::endl;  // might look like: group_offset = (next/3)*729*3 +
                           // (next%3)*729;
      } else {
        // if smaller_dim is 2187 > 1024 this should look like size_t
        // group_offset = (next/3)*2187*3 + (next%3)*2187 + (group_id % 3)*729;
        StockhamGenerator::hcKernWrite(transKernel, 6)
            << "group_offset = (next/" << dim_ratio << ")*" << smaller_dim
            << "*" << dim_ratio << " + (next%" << dim_ratio << ")*"
            << smaller_dim << " + (group_id % " << WG_per_line << ")*"
            << LDS_per_WG << ";" << std::endl;
      }
    } else {
      StockhamGenerator::hcKernWrite(transKernel, 6) << "next = (prev*" << dim_ratio << ")%"
                                  << smaller_dim * dim_ratio - 1 << ";"
                                  << std::endl;
      // takes care the last row
      StockhamGenerator::hcKernWrite(transKernel, 6)
          << "if (prev == " << smaller_dim * dim_ratio - 1 << ")" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "next = " << smaller_dim * dim_ratio - 1
                                  << ";" << std::endl;
      if (WG_per_line == 1)  // might look like: size_t group_offset = prev*729;
        StockhamGenerator::hcKernWrite(transKernel, 6) << "group_offset = (next*" << smaller_dim
                                    << ");" << std::endl;
      else  // if smaller_dim is 2187 > 1024 this should look like size_t
            // group_offset = next*2187 + (group_id % 3)*729;
        StockhamGenerator::hcKernWrite(transKernel, 6) << "group_offset = (next*" << smaller_dim
                                    << ") + (group_id % " << WG_per_line << ")*"
                                    << LDS_per_WG << ";" << std::endl;
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    switch (params.fft_inputLayout) {
      case HCFFT_REAL:
      case HCFFT_COMPLEX_INTERLEAVED: {
        for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
          if (i + 256 < LDS_per_WG) {
            if (twiddleSwapKernelIn) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.y + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].y * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].x * twiddle_factor.y;" << std::endl;
              }
            } else {
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "nextValue[idx+" << i
                  << "] = inputA[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            }
          } else {
            // need to handle boundary
            StockhamGenerator::hcKernWrite(transKernel, 6) << "if(idx+" << i << "<" << LDS_per_WG
                                        << "){" << std::endl;
            if (twiddleSwapKernelIn) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.y + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA[inOffset + group_offset+idx+" << i
                    << "].x * twiddle_factor.x + inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA[inOffset + group_offset+idx+" << i
                    << "].y * twiddle_factor.x - inputA[inOffset + "
                       "group_offset+idx+"
                    << i << "].x * twiddle_factor.y;" << std::endl;
              }
            } else {
              StockhamGenerator::hcKernWrite(transKernel, 9)
                  << "nextValue[idx+" << i
                  << "] = inputA[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
            }
          }
        }
        break;
      }
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
  case HCFFT_COMPLEX_PLANAR: {
    for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
      if (i + 256 < LDS_per_WG) {
        if (twiddleSwapKernelIn) {
          if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                    // have a tall or wide
                                                    // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
          } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.y + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA_I[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_R[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
              }
        } else {
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "nextValue[idx+" << i
                  << "].x = inputA_R[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "nextValue[idx+" << i
                  << "].y = inputA_I[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            }
      } else {
            // need to handle boundary
            StockhamGenerator::hcKernWrite(transKernel, 6) << "if(idx+" << i << "<" << LDS_per_WG
                                        << "){" << std::endl;
            if (twiddleSwapKernelIn) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide; read input index realted
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.y + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].x = inputA_R[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x + inputA_I[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 3)
                    << "nextValue[idx+" << i
                    << "].y = inputA_I[inOffset + group_offset+idx+" << i
                    << "] * twiddle_factor.x - inputA_R[inOffset + "
                       "group_offset+idx+"
                    << i << "] * twiddle_factor.y;" << std::endl;
              }
            } else {
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "nextValue[idx+" << i
                  << "].x = inputA_R[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "nextValue[idx+" << i
                  << "].y = inputA_I[inOffset + group_offset+idx+" << i << "];"
                  << std::endl;
            }
            StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
          }
        }
        break;
      }
      default:
        return HCFFT_INVALID;
    }

    StockhamGenerator::hcKernWrite(transKernel, 6) << "tidx.barrier.wait();" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    switch (params.fft_inputLayout) {
      case HCFFT_REAL:  // for real case this is different
      case HCFFT_COMPLEX_INTERLEAVED: {
        if (twiddleSwapKernelOut) {
          StockhamGenerator::hcKernWrite(transKernel, 6) << "size_t p;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6) << "size_t q;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6) << dtComplex << " twiddle_factor;"
                                      << std::endl;

          for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
            if (i + 256 < LDS_per_WG) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].x = prevValue[idx+" << i
                    << "].x * twiddle_factor.x - prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].y = prevValue[idx+" << i
                    << "].x * twiddle_factor.y + prevValue[idx+" << i
                    << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].x = prevValue[idx+" << i
                    << "].x * twiddle_factor.x + prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].y = prevValue[idx+" << i
                    << "].y * twiddle_factor.x - prevValue[idx+" << i
                    << "].x * twiddle_factor.y;" << std::endl;
              }
              // StockhamGenerator::hcKernWrite(transKernel, 6) << "inputA[inOffset +
              // group_offset+idx+" << i << "] = prevValue[idx+" << i << "];" <<
              // std::endl;
            } else {
              // need to handle boundary
              StockhamGenerator::hcKernWrite(transKernel, 6) << "if(idx+" << i << "<" << LDS_per_WG
                                          << "){" << std::endl;
              if (params.fft_N[0] > params.fft_N[1])  {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].x = prevValue[idx+" << i
                    << "].x * twiddle_factor.x - prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].y = prevValue[idx+" << i
                    << "].x * twiddle_factor.y + prevValue[idx+" << i
                    << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].x = prevValue[idx+" << i
                    << "].x * twiddle_factor.x + prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA[inOffset + group_offset+idx+" << i
                    << "].y = prevValue[idx+" << i
                    << "].y * twiddle_factor.x - prevValue[idx+" << i
                    << "].x * twiddle_factor.y;" << std::endl;
              }
              // StockhamGenerator::hcKernWrite(transKernel, 9) << "inputA[inOffset +
              // group_offset+idx+" << i << "] = prevValue[idx+" << i << "];" <<
              // std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
            }
          }
        } else if (!twiddleSwapKernelOut)  {  // could be twiddleSwapKernelIn
          for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
            if (i + 256 < LDS_per_WG) {
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "inputA[inOffset + group_offset+idx+" << i
                  << "] = prevValue[idx+" << i << "];" << std::endl;
            } else {
              // need to handle boundary
              StockhamGenerator::hcKernWrite(transKernel, 6) << "if(idx+" << i << "<" << LDS_per_WG
                                          << "){" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9)
                  << "inputA[inOffset + group_offset+idx+" << i
                  << "] = prevValue[idx+" << i << "];" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
            }
          }
        }
        break;
      }
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      case HCFFT_COMPLEX_PLANAR: {
        if (twiddleSwapKernelOut) {
          StockhamGenerator::hcKernWrite(transKernel, 6) << "size_t p;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6) << "size_t q;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6) << dtComplex << " twiddle_factor;"
                                      << std::endl;
          for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
            if (i + 256 < LDS_per_WG) {
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_R[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].x * twiddle_factor.x - prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_I[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].x * twiddle_factor.y + prevValue[idx+" << i
                    << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_R[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].x * twiddle_factor.x + prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_I[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].y * twiddle_factor.x - prevValue[idx+" << i
                    << "].x * twiddle_factor.y;" << std::endl;
              }
            } else {
              // need to handle boundary
              StockhamGenerator::hcKernWrite(transKernel, 6) << "if(idx+" << i << "<" << LDS_per_WG
                                          << "){" << std::endl;
              if (params.fft_N[0] > params.fft_N[1]) {  // decides whether we
                                                      // have a tall or wide
                                                      // rectangle
                // input is wide; output is tall
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << smaller_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << smaller_dim << ";"
                                            << std::endl;
              } else {
                // input is tall; output is wide
                StockhamGenerator::hcKernWrite(transKernel, 6) << "p = (group_offset+idx+" << i
                                            << ")/" << bigger_dim << ";"
                                            << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6) << "q = (group_offset+idx+" << i
                                            << ")%" << bigger_dim << ";"
                                            << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "twiddle_factor = TW3step"
                                          << plHandle << "(p*q" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ", ";
              StockhamGenerator::hcKernWrite(transKernel, 9) << StockhamGenerator::TwTableLargeName() << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 9) << ");" << std::endl;
              if (fwd) {
                // forward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_R[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].x * twiddle_factor.x - prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_I[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].x * twiddle_factor.y + prevValue[idx+" << i
                    << "].y * twiddle_factor.x;" << std::endl;
              } else {
                // backward
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_R[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].x * twiddle_factor.x + prevValue[idx+" << i
                    << "].y * twiddle_factor.y;" << std::endl;
                StockhamGenerator::hcKernWrite(transKernel, 6)
                    << "inputA_I[inOffset + group_offset+idx+" << i
                    << "] = prevValue[idx+" << i
                    << "].y * twiddle_factor.x - prevValue[idx+" << i
                    << "].x * twiddle_factor.y;" << std::endl;
              }
              StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
            }
            StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
          }
        } else if (!twiddleSwapKernelOut) {  // could be twiddleSwapKernelIn
          for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
            if (i + 256 < LDS_per_WG) {
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "inputA_R[inOffset + group_offset+idx+" << i
                  << "] = prevValue[idx+" << i << "].x;" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "inputA_I[inOffset + group_offset+idx+" << i
                  << "] = prevValue[idx+" << i << "].y;" << std::endl;
            } else {
              // need to handle boundary
              StockhamGenerator::hcKernWrite(transKernel, 6) << "if(idx+" << i << "<" << LDS_per_WG
                                          << "){" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "inputA_R[inOffset + group_offset+idx+" << i
                  << "] = prevValue[idx+" << i << "].x;" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6)
                  << "inputA_I[inOffset + group_offset+idx+" << i
                  << "] = prevValue[idx+" << i << "].y;" << std::endl;
              StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
            }
          }
        }
        break;
      }
      default:
        return HCFFT_INVALID;
    }
    StockhamGenerator::hcKernWrite(transKernel, 6) << "tidx.barrier.wait();" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    switch (params.fft_inputLayout) {
      case HCFFT_REAL:
      case HCFFT_COMPLEX_INTERLEAVED:
      case HCFFT_COMPLEX_PLANAR: {
        for (size_t i = 0; i < LDS_per_WG; i = i + 256) {
          if (i + 256 < LDS_per_WG) {
            StockhamGenerator::hcKernWrite(transKernel, 6) << "prevValue[idx+" << i
                                        << "] = nextValue[idx+" << i << "];"
                                        << std::endl;
          } else {
            // need to handle boundary
            StockhamGenerator::hcKernWrite(transKernel, 6) << "if(idx+" << i << "<" << LDS_per_WG
                                        << "){" << std::endl;
            StockhamGenerator::hcKernWrite(transKernel, 9) << "prevValue[idx + " << i
                                        << "] = nextValue[idx + " << i << "]; "
                                        << std::endl;
            StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
          }
        }
        break;
      }
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      default:
        return HCFFT_INVALID;
    }

    StockhamGenerator::hcKernWrite(transKernel, 6) << "tidx.barrier.wait();" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "prev = next;" << std::endl;
    if (WG_per_line == 1)
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}while(next!=swap_table[group_id][0]);"
                                  << std::endl;  // end of do-while
    else
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}while(next!=swap_table[group_id/"
                                  << WG_per_line << "][0]);"
                                  << std::endl;  // end of do-while
    StockhamGenerator::hcKernWrite(transKernel, 0) << "}).wait();\n}}\n"
                                << std::endl;  // end of kernel

    if (!twiddleSwapKernel)
      break;  // break for bothDir only need one kernel if twiddle is not done
              // here
  }  // end of for (size_t bothDir = 0; bothDir < 2; bothDir++)

  // by now the kernel string is generated
  strKernel = transKernel.str();
  return HCFFT_SUCCEEDS;
}

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
    std::vector<size_t> gWorkSize, std::vector<size_t> lWorkSize,
    size_t count) {
  strKernel.reserve(4096);
  std::stringstream transKernel(std::stringstream::out);

  // These strings represent the various data types we read or write in the
  // kernel, depending on how the plan
  // is configured
  std::string dtInput;    // The type read as input into kernel
  std::string dtOutput;   // The type written as output from kernel
  std::string dtPlanar;   // Fundamental type for planar arrays
  std::string dtComplex;  // Fundamental type for complex arrays

  switch (params.fft_precision) {
    case HCFFT_SINGLE:
      dtPlanar = "float";
      dtComplex = "float_2";
      break;
    case HCFFT_DOUBLE:
      dtPlanar = "double";
      dtComplex = "double2";
      break;
    default:
      return HCFFT_INVALID;
      break;
  }

  //  it is a better idea to do twiddle in swap kernel if we will have a swap
  //  kernel.
  //  for pure square transpose, twiddle will be done in transpose kernel
  bool twiddleTransposeKernel =
      params.fft_3StepTwiddle &&
      (params.transposeMiniBatchSize == 1);  // when transposeMiniBatchSize == 1
                                             // it is guaranteed to be a sqaure
                                             // matrix transpose
  // If twiddle computation has been requested, generate the lookup function

  if (twiddleTransposeKernel) {
    std::string str;
    if (params.fft_precision == HCFFT_SINGLE) {
      StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2,
                                           StockhamGenerator::P_SINGLE>
          twLarge(params.fft_N[0] * params.fft_N[1]);
      twLarge.GenerateTwiddleTable(str, plHandle);
      twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
    } else {
      StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2,
                                           StockhamGenerator::P_DOUBLE>
          twLarge(params.fft_N[0] * params.fft_N[1]);
      twLarge.GenerateTwiddleTable(str, plHandle);
      twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
    }
    StockhamGenerator::hcKernWrite(transKernel, 0) << str << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 0) << std::endl;
  }

  // This detects whether the input matrix is square
  bool notSquare = (params.fft_N[0] == params.fft_N[1]) ? false : true;

  if (notSquare && (params.fft_placeness == HCFFT_INPLACE))
    return HCFFT_INVALID;

  // This detects whether the input matrix is a multiple of 16*reshapefactor or
  // not

  bool mult_of_16 =
      (params.fft_N[0] % (reShapeFactor * 16) == 0) ? true : false;

  for (size_t bothDir = 0; bothDir < 2; bothDir++) {
    bool fwd = bothDir ? false : true;

    std::string funcName;
    if (twiddleTransposeKernel)  // it makes more sense to do twiddling in swap
                                 // kernel
      funcName = fwd ? "transpose_square_tw_fwd" : "transpose_square_tw_back";
    else
      funcName = "transpose_square";
    funcName += SztToStr(count);

    // Generate kernel API
    genTransposePrototype(params, lwSize, dtPlanar, dtComplex, funcName,
                          transKernel, dtInput, dtOutput,
                          twiddleTransposeKernel);
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::extent<2> grdExt( ";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(gWorkSize[0]) << ", 1); \n"
                                << "\thc::tiled_extent<2> t_ext = grdExt.tile(";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(lwSize) << ", 1);\n";
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::parallel_for_each(acc_view, t_ext, "
                                   "[=] (hc::tiled_index<2> tidx) [[hc]]\n\t "
                                   "{ ";

    size_t wgPerBatch;
    if (mult_of_16)
      wgPerBatch = (params.fft_N[0] / 16 / reShapeFactor) *
                   (params.fft_N[0] / 16 / reShapeFactor + 1) / 2;
    else
      wgPerBatch = (params.fft_N[0] / (16 * reShapeFactor) + 1) *
                   (params.fft_N[0] / (16 * reShapeFactor) + 1 + 1) / 2;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t numGroupsY_1 = " << wgPerBatch
                                << ";" << std::endl;

    for (size_t i = 2; i < params.fft_DataDim - 1; i++) {
      StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t numGroupsY_" << i
                                  << " = numGroupsY_" << i - 1 << " * "
                                  << params.fft_N[i] << ";" << std::endl;
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t g_index;" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;

    OffsetCalculation(transKernel, params, true);

    if (params.fft_placeness == HCFFT_OUTOFPLACE)
      OffsetCalculation(transKernel, params, false);

    if (params.fft_placeness == HCFFT_INPLACE) {
      switch (params.fft_inputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 3) << dtInput << " *outputA = inputA;"
                                      << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          StockhamGenerator::hcKernWrite(transKernel, 3) << dtInput << " *outputA_R = inputA_R;"
                                      << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 3) << dtInput << " *outputA_I = inputA_I;"
                                      << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;

    // Now compute the corresponding y,x coordinates
    // for a triangular indexing
    if (mult_of_16)
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "float row = (" << -2.0f * params.fft_N[0] / 16 / reShapeFactor - 1
          << "+sqrt(("
          << 4.0f * params.fft_N[0] / 16 / reShapeFactor *
                 (params.fft_N[0] / 16 / reShapeFactor + 1)
          << "-8.0f*g_index- 7)))/ (-2.0f);" << std::endl;
    else
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "float row = ("
          << -2.0f * (params.fft_N[0] / (16 * reShapeFactor) + 1) - 1
          << "+sqrt(("
          << 4.0f * (params.fft_N[0] / (16 * reShapeFactor) + 1) *
                 (params.fft_N[0] / (16 * reShapeFactor) + 1 + 1)
          << "-8.0f*g_index- 7)))/ (-2.0f);" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "if (row == (float)(size_t)row) row -= 1; "
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t t_gy = (size_t)row;"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    if (mult_of_16)
      StockhamGenerator::hcKernWrite(transKernel, 3) << "const long t_gx_p = g_index - "
                                  << (params.fft_N[0] / 16 / reShapeFactor)
                                  << "*t_gy + t_gy*(t_gy + 1) / 2;"
                                  << std::endl;
    else
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "const long t_gx_p = g_index - "
          << (params.fft_N[0] / (16 * reShapeFactor) + 1)
          << "*t_gy + t_gy*(t_gy + 1) / 2;" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const long t_gy_p = t_gx_p - t_gy;"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t d_lidx = tidx.local[0] % 16;"
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t d_lidy = tidx.local[0] / 16;"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "const size_t lidy = (d_lidy * 16 + d_lidx) /"
        << (16 * reShapeFactor) << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "const size_t lidx = (d_lidy * 16 + d_lidx) %"
        << (16 * reShapeFactor) << ";" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t idx = lidx + t_gx_p*"
                                << 16 * reShapeFactor << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t idy = lidy + t_gy_p*"
                                << 16 * reShapeFactor << ";" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t starting_index_yx = t_gy_p*"
                                << 16 * reShapeFactor << " + t_gx_p*"
                                << 16 * reShapeFactor * params.fft_N[0] << ";"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtComplex << " xy_s["
                                << 16 * reShapeFactor * 16 * reShapeFactor
                                << "];" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtComplex << " yx_s["
                                << 16 * reShapeFactor * 16 * reShapeFactor
                                << "];" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << dtComplex << " tmpm, tmpt;" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    // Step 1: Load both blocks into local memory
    // Here I load inputA for both blocks contiguously and write it contigously
    // into
    // the corresponding shared memories.
    // Afterwards I use non-contiguous access from local memory and write
    // contiguously
    // back into the arrays

    if (mult_of_16) {
      StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t index;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "index = lidy*" << 16 * reShapeFactor
                                  << " + lidx + loop*256;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_inputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED: {
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpm = inputA[iOffset + (idy + loop *" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpt = inputA[iOffset + (lidy + loop *" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];"
              << std::endl;
        } break;
        case HCFFT_COMPLEX_PLANAR:
          dtInput = dtPlanar;
          dtOutput = dtPlanar;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpm.x = inputA_R[iOffset + (idy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpm.y = inputA_I[iOffset + (idy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;

          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpt.x = inputA_R[iOffset + (lidy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpt.y = inputA_I[iOffset + (lidy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      // it makes more sense to do twiddling in swap kernel
      // If requested, generate the Twiddle math to multiply constant values
      if (twiddleTransposeKernel)
        genTwiddleMath(plHandle, params, transKernel, dtComplex, fwd);

      StockhamGenerator::hcKernWrite(transKernel, 6) << "xy_s[index] = tmpm; " << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "yx_s[index] = tmpt; " << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "tidx.barrier.wait();" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

      // Step2: Write from shared to global
      StockhamGenerator::hcKernWrite(transKernel, 3) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "index = lidx*" << 16 * reShapeFactor
                                  << " + lidy + " << 16 / reShapeFactor
                                  << "*loop;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_outputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA[iOffset + (idy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + idx] = yx_s[index];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA[iOffset + (lidy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + lidx+ starting_index_yx] = xy_s[index];"
              << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_R[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x;"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_I[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y;"
              << std::endl;

          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_R[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx+ starting_index_yx] = xy_s[index].x;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_I[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx+ starting_index_yx] = xy_s[index].y;" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;
    } else {  // mult_of_16
      StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t index;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "if (" << params.fft_N[0]
                                  << " - (t_gx_p + 1) *" << 16 * reShapeFactor
                                  << ">0){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidy*" << 16 * reShapeFactor
                                  << " + lidx + loop*256;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_inputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpm = inputA[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpt = inputA[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];"
              << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          dtInput = dtPlanar;
          dtOutput = dtPlanar;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpm.x = inputA_R[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpm.y = inputA_I[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;

          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpt.x = inputA_R[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpt.y = inputA_I[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      // it makes more sense to do twiddling in swap kernel
      // If requested, generate the Twiddle math to multiply constant values
      if (twiddleTransposeKernel)
        genTwiddleMath(plHandle, params, transKernel, dtComplex, fwd);

      StockhamGenerator::hcKernWrite(transKernel, 9) << "xy_s[index] = tmpm;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "yx_s[index] = tmpt;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "else{" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidy*" << 16 * reShapeFactor
                                  << " + lidx + loop*256;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_inputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << "&& idx<" << params.fft_N[0] << ")"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpm = inputA[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p *" << 16 * reShapeFactor << " + lidx)<"
              << params.fft_N[0] << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << ") " << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpt = inputA[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];"
              << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          dtInput = dtPlanar;
          dtOutput = dtPlanar;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << "&& idx<" << params.fft_N[0] << ") {"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpm.x = inputA_R[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpm.y = inputA_I[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx]; }"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p *" << 16 * reShapeFactor << " + lidx)<"
              << params.fft_N[0] << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << ") {" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpt.x = inputA_R[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpt.y = inputA_I[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx]; }" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      // If requested, generate the Twiddle math to multiply constant values
      if (twiddleTransposeKernel)
        genTwiddleMath(plHandle, params, transKernel, dtComplex, fwd);

      StockhamGenerator::hcKernWrite(transKernel, 9) << "xy_s[index] = tmpm;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "yx_s[index] = tmpt;" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 9) << "}" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "tidx.barrier.wait();" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

      // Step2: Write from shared to global

      StockhamGenerator::hcKernWrite(transKernel, 3) << "if (" << params.fft_N[0]
                                  << " - (t_gx_p + 1) *" << 16 * reShapeFactor
                                  << ">0){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor
                                  << " + lidy + " << 16 / reShapeFactor
                                  << "*loop ;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_outputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA[iOffset + (idy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + idx] = yx_s[index];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA[iOffset + (lidy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index]; " << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_R[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x;"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_I[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y;"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_R[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].x; " << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_I[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].y; " << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "else{" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor
                                  << " + lidy + " << 16 / reShapeFactor
                                  << "*loop;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_outputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << " && idx<" << params.fft_N[0] << ")"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA[iOffset + (idy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + idx] = yx_s[index]; " << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p * " << 16 * reShapeFactor << " + lidx)<"
              << params.fft_N[0] << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << ")" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA[iOffset + (lidy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index];" << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << " && idx<" << params.fft_N[0] << ") {"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_R[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x; "
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_I[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y; }"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p * " << 16 * reShapeFactor << " + lidx)<"
              << params.fft_N[0] << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<"
              << params.fft_N[0] << ") {" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_R[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].x;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_I[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].y; }" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;  // end for
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;  // end else
    }
    StockhamGenerator::hcKernWrite(transKernel, 0) << "}).wait();\n}}\n" << std::endl;

    strKernel = transKernel.str();

    if (!twiddleTransposeKernel) break;  // break for bothDir
  }
  return HCFFT_SUCCEEDS;
}

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
    std::vector<size_t> gWorkSize, std::vector<size_t> lWorkSize,
    size_t count) {
  strKernel.reserve(4096);
  std::stringstream transKernel(std::stringstream::out);

  // These strings represent the various data types we read or write in the
  // kernel, depending on how the plan
  // is configured
  std::string dtInput;    // The type read as input into kernel
  std::string dtOutput;   // The type written as output from kernel
  std::string dtPlanar;   // Fundamental type for planar arrays
  std::string dtComplex;  // Fundamental type for complex arrays
  bool genTwiddle = false;

  switch (params.fft_precision) {
    case HCFFT_SINGLE:
      dtPlanar = "float";
      dtComplex = "float_2";
      break;
    case HCFFT_DOUBLE:
      dtPlanar = "double";
      dtComplex = "double2";
      break;
    default:
      return HCFFT_INVALID;
      break;
  }

  // If twiddle computation has been requested, generate the lookup function
  if (params.fft_3StepTwiddle) {
    genTwiddle = true;
    std::string str;
    if (params.fft_precision == HCFFT_SINGLE) {
      StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2,
                                           StockhamGenerator::P_SINGLE>
          twLarge(params.fft_N[0] * params.fft_N[1]);
      twLarge.GenerateTwiddleTable(str, plHandle);
      twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
    } else {
      StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2,
                                           StockhamGenerator::P_DOUBLE>
          twLarge(params.fft_N[0] * params.fft_N[1]);
      twLarge.GenerateTwiddleTable(str, plHandle);
      twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
    }
    StockhamGenerator::hcKernWrite(transKernel, 0) << str << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 0) << std::endl;
  }

  size_t smaller_dim =
      (params.fft_N[0] < params.fft_N[1]) ? params.fft_N[0] : params.fft_N[1];
  size_t bigger_dim =
      (params.fft_N[0] >= params.fft_N[1]) ? params.fft_N[0] : params.fft_N[1];
  size_t dim_ratio = bigger_dim / smaller_dim;

  // This detects whether the input matrix is rectangle of ratio 1:2

  if ((params.fft_N[0] != 2 * params.fft_N[1]) &&
      (params.fft_N[1] != 2 * params.fft_N[0]) &&
      (params.fft_N[0] != 3 * params.fft_N[1]) &&
      (params.fft_N[1] != 3 * params.fft_N[0]) &&
      (params.fft_N[0] != 5 * params.fft_N[1]) &&
      (params.fft_N[1] != 5 * params.fft_N[0]) &&
      (params.fft_N[0] != 10 * params.fft_N[1]) &&
      (params.fft_N[1] != 10 * params.fft_N[0])) {
    return HCFFT_INVALID;
  }

  if (params.fft_placeness == HCFFT_OUTOFPLACE) {
    return HCFFT_INVALID;
  }

  // This detects whether the input matrix is a multiple of 16*reshapefactor or
  // not

  bool mult_of_16 = (smaller_dim % (reShapeFactor * 16) == 0) ? true : false;

  for (size_t bothDir = 0; bothDir < 2; bothDir++) {
    bool fwd = bothDir ? false : true;

    std::string funcName;
    if (params.fft_3StepTwiddle) {
      funcName =
          fwd ? "transpose_nonsquare_tw_fwd" : "transpose_nonsquare_tw_back";
    } else {
      funcName = "transpose_nonsquare";
    }

    funcName += SztToStr(count);
    // Generate kernel API
    genTransposePrototypeLeadingDimensionBatched(
        params, lwSize, dtPlanar, dtComplex, funcName, transKernel, dtInput,
        dtOutput, genTwiddle);
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::extent<2> grdExt( ";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(gWorkSize[0]) << ", 1); \n"
                                << "\thc::tiled_extent<2> t_ext = grdExt.tile(";
    StockhamGenerator::hcKernWrite(transKernel, 3) << SztToStr(lwSize) << ", 1);\n";
    StockhamGenerator::hcKernWrite(transKernel, 3) << "\thc::parallel_for_each(acc_view, t_ext, "
                                   "[=] (hc::tiled_index<2> tidx) [[hc]]\n\t "
                                   "{ ";

    if (mult_of_16)  // number of WG per sub square block
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "const size_t  numGroups_square_matrix_Y_1 = "
          << (smaller_dim / 16 / reShapeFactor) *
                 (smaller_dim / 16 / reShapeFactor + 1) / 2
          << ";" << std::endl;
    else
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "const size_t  numGroups_square_matrix_Y_1 = "
          << (smaller_dim / (16 * reShapeFactor) + 1) *
                 (smaller_dim / (16 * reShapeFactor) + 1 + 1) / 2
          << ";" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "const size_t  numGroupsY_1 =  numGroups_square_matrix_Y_1 * "
        << dim_ratio << ";" << std::endl;

    for (size_t i = 2; i < params.fft_DataDim - 1; i++) {
      StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t numGroupsY_" << i
                                  << " = numGroupsY_" << i - 1 << " * "
                                  << params.fft_N[i] << ";" << std::endl;
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t g_index;" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t square_matrix_index;" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t square_matrix_offset;" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;

    OffsetCalcLeadingDimensionBatched(transKernel, params);

    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "square_matrix_index = (g_index / numGroups_square_matrix_Y_1) ;"
        << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "g_index = g_index % numGroups_square_matrix_Y_1"
        << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;

    if (smaller_dim == params.fft_N[1]) {
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "square_matrix_offset = square_matrix_index * " << smaller_dim
          << ";" << std::endl;
    } else {
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "square_matrix_offset = square_matrix_index *"
          << smaller_dim * smaller_dim << ";" << std::endl;
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << "iOffset += square_matrix_offset ;"
                                << std::endl;

    switch (params.fft_inputLayout) {
      case HCFFT_COMPLEX_INTERLEAVED:
      case HCFFT_REAL:
        StockhamGenerator::hcKernWrite(transKernel, 3) << dtInput << " *outputA = inputA;"
                                    << std::endl;
        break;
      case HCFFT_COMPLEX_PLANAR:
        StockhamGenerator::hcKernWrite(transKernel, 3) << dtInput << " *outputA_R = inputA_R;"
                                    << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 3) << dtInput << " *outputA_I = inputA_I;"
                                    << std::endl;
        break;
      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;
      default:
        return HCFFT_INVALID;
    }

    StockhamGenerator::hcKernWrite(transKernel, 3) << std::endl;

    // Now compute the corresponding y,x coordinates
    // for a triangular indexing
    if (mult_of_16)
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "float row = (" << -2.0f * smaller_dim / 16 / reShapeFactor - 1
          << "+sqrt(("
          << 4.0f * smaller_dim / 16 / reShapeFactor *
                 (smaller_dim / 16 / reShapeFactor + 1)
          << "-8.0f*g_index- 7)))/ (-2.0f);" << std::endl;
    else
      StockhamGenerator::hcKernWrite(transKernel, 3)
          << "float row = ("
          << -2.0f * (smaller_dim / (16 * reShapeFactor) + 1) - 1 << "+sqrt(("
          << 4.0f * (smaller_dim / (16 * reShapeFactor) + 1) *
                 (smaller_dim / (16 * reShapeFactor) + 1 + 1)
          << "-8.0f*g_index- 7)))/ (-2.0f);" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "if (row == (float)(int)row) row -= 1; "
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t t_gy = (int)row;" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    if (mult_of_16)
      StockhamGenerator::hcKernWrite(transKernel, 3) << "const long t_gx_p = g_index - "
                                  << (smaller_dim / 16 / reShapeFactor)
                                  << "*t_gy + t_gy*(t_gy + 1) / 2;"
                                  << std::endl;
    else
      StockhamGenerator::hcKernWrite(transKernel, 3) << "const long t_gx_p = g_index - "
                                  << (smaller_dim / (16 * reShapeFactor) + 1)
                                  << "*t_gy + t_gy*(t_gy + 1) / 2;"
                                  << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const long t_gy_p = t_gx_p - t_gy;"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t d_lidx = tidx.local[0] % 16;"
                                << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t d_lidy = tidx.local[0] / 16;"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "const size_t lidy = (d_lidy * 16 + d_lidx) /"
        << (16 * reShapeFactor) << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3)
        << "const size_t lidx = (d_lidy * 16 + d_lidx) %"
        << (16 * reShapeFactor) << ";" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t idx = lidx + t_gx_p*"
                                << 16 * reShapeFactor << ";" << std::endl;
    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t idy = lidy + t_gy_p*"
                                << 16 * reShapeFactor << ";" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "const size_t starting_index_yx = t_gy_p*"
                                << 16 * reShapeFactor << " + t_gx_p*"
                                << 16 * reShapeFactor * params.fft_N[0] << ";"
                                << std::endl;

    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    switch (params.fft_inputLayout) {
      case HCFFT_REAL:
      case HCFFT_COMPLEX_INTERLEAVED:
        StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtInput << " xy_s["
                                    << 16 * reShapeFactor * 16 * reShapeFactor
                                    << "];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtInput << " yx_s["
                                    << 16 * reShapeFactor * 16 * reShapeFactor
                                    << "];" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 3) << dtInput << " tmpm, tmpt;" << std::endl;
        break;
      case HCFFT_COMPLEX_PLANAR:
        StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtComplex << " xy_s["
                                    << 16 * reShapeFactor * 16 * reShapeFactor
                                    << "];" << std::endl;
        StockhamGenerator::hcKernWrite(transKernel, 3) << "tile_static " << dtComplex << " yx_s["
                                    << 16 * reShapeFactor * 16 * reShapeFactor
                                    << "];" << std::endl;

        StockhamGenerator::hcKernWrite(transKernel, 3) << dtComplex << " tmpm, tmpt;" << std::endl;
        break;
      default:
        return HCFFT_INVALID;
    }
    StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

    // Step 1: Load both blocks into local memory
    // Here I load inputA for both blocks contiguously and write it contigously
    // into
    // the corresponding shared memories.
    // Afterwards I use non-contiguous access from local memory and write
    // contiguously
    // back into the arrays

    if (mult_of_16) {
      StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t index;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "index = lidy*" << 16 * reShapeFactor
                                  << " + lidx + loop*256;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_inputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
        case HCFFT_REAL: {
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpm = inputA[iOffset + (idy + loop *" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpt = inputA[iOffset + (lidy + loop *" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];"
              << std::endl;
        } break;
        case HCFFT_COMPLEX_PLANAR:
          dtInput = dtPlanar;
          dtOutput = dtPlanar;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpm.x = inputA_R[iOffset + (idy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpm.y = inputA_I[iOffset + (idy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;

          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpt.x = inputA_R[iOffset + (lidy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "tmpt.y = inputA_I[iOffset + (lidy + loop *"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        default:
          return HCFFT_INVALID;
      }

      // If requested, generate the Twiddle math to multiply constant values
      if (params.fft_3StepTwiddle)
        genTwiddleMathLeadingDimensionBatched(plHandle, params, transKernel,
                                              dtComplex, fwd);

      StockhamGenerator::hcKernWrite(transKernel, 6) << "xy_s[index] = tmpm; " << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "yx_s[index] = tmpt; " << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "tidx.barrier.wait();" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

      // Step2: Write from shared to global
      StockhamGenerator::hcKernWrite(transKernel, 3) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "index = lidx*" << 16 * reShapeFactor
                                  << " + lidy + " << 16 / reShapeFactor
                                  << "*loop;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_outputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA[iOffset + (idy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + idx] = yx_s[index];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA[iOffset + (lidy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + lidx+ starting_index_yx] = xy_s[index];"
              << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_R[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x;"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_I[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y;"
              << std::endl;

          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_R[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx+ starting_index_yx] = xy_s[index].x;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 6)
              << "outputA_I[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx+ starting_index_yx] = xy_s[index].y;" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;
    } else {
      StockhamGenerator::hcKernWrite(transKernel, 3) << "size_t index;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "if (" << smaller_dim
                                  << " - (t_gx_p + 1) *" << 16 * reShapeFactor
                                  << ">0){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidy*" << 16 * reShapeFactor
                                  << " + lidx + loop*256;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_inputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
        case HCFFT_REAL:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpm = inputA[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpt = inputA[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];"
              << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          dtInput = dtPlanar;
          dtOutput = dtPlanar;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpm.x = inputA_R[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpm.y = inputA_I[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;

          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpt.x = inputA_R[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "tmpt.y = inputA_I[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        default:
          return HCFFT_INVALID;
      }

      // If requested, generate the Twiddle math to multiply constant values
      if (params.fft_3StepTwiddle)
        genTwiddleMathLeadingDimensionBatched(plHandle, params, transKernel,
                                              dtComplex, fwd);

      StockhamGenerator::hcKernWrite(transKernel, 9) << "xy_s[index] = tmpm;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "yx_s[index] = tmpt;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "else{" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidy*" << 16 * reShapeFactor
                                  << " + lidx + loop*256;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_inputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
        case HCFFT_REAL:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << "&& idx<" << smaller_dim << ")" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpm = inputA[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p *" << 16 * reShapeFactor << " + lidx)<"
              << smaller_dim << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << ") " << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpt = inputA[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];"
              << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          dtInput = dtPlanar;
          dtOutput = dtPlanar;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << "&& idx<" << smaller_dim << ") {" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpm.x = inputA_R[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpm.y = inputA_I[iOffset + (idy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx]; }"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p *" << 16 * reShapeFactor << " + lidx)<"
              << smaller_dim << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << ") {" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpt.x = inputA_R[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "tmpt.y = inputA_I[iOffset + (lidy + loop*"
              << 16 / reShapeFactor << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx]; }" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        default:
          return HCFFT_INVALID;
      }

      // If requested, generate the Twiddle math to multiply constant values
      if (params.fft_3StepTwiddle)
        genTwiddleMathLeadingDimensionBatched(plHandle, params, transKernel,
                                              dtComplex, fwd);

      StockhamGenerator::hcKernWrite(transKernel, 9) << "xy_s[index] = tmpm;" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "yx_s[index] = tmpt;" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 9) << "}" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "tidx.barrier.wait();" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "" << std::endl;

      // Step2: Write from shared to global

      StockhamGenerator::hcKernWrite(transKernel, 3) << "if (" << smaller_dim
                                  << " - (t_gx_p + 1) *" << 16 * reShapeFactor
                                  << ">0){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor
                                  << " + lidy + " << 16 / reShapeFactor
                                  << "*loop ;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_outputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA[iOffset + (idy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + idx] = yx_s[index];" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA[iOffset + (lidy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index]; " << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_R[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x;"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_I[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y;"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_R[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].x; " << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "outputA_I[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].y; " << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 3) << "else{" << std::endl;
      StockhamGenerator::hcKernWrite(transKernel, 6) << "for (size_t loop = 0; loop<"
                                  << reShapeFactor * reShapeFactor
                                  << "; ++loop){" << std::endl;

      StockhamGenerator::hcKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor
                                  << " + lidy + " << 16 / reShapeFactor
                                  << "*loop;" << std::endl;

      // Handle planar and interleaved right here
      switch (params.fft_outputLayout) {
        case HCFFT_COMPLEX_INTERLEAVED:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << " && idx<" << smaller_dim << ")" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA[iOffset + (idy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0] << " + idx] = yx_s[index]; " << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p * " << 16 * reShapeFactor << " + lidx)<"
              << smaller_dim << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << ")" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA[iOffset + (lidy + loop*" << 16 / reShapeFactor << ")*"
              << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index];" << std::endl;
          break;
        case HCFFT_COMPLEX_PLANAR:
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((idy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << " && idx<" << smaller_dim << ") {" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_R[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x; "
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_I[iOffset + (idy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y; }"
              << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 9)
              << "if ((t_gy_p * " << 16 * reShapeFactor << " + lidx)<"
              << smaller_dim << " && (t_gx_p * " << 16 * reShapeFactor
              << " + lidy + loop*" << 16 / reShapeFactor << ")<" << smaller_dim
              << ") {" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_R[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].x;" << std::endl;
          StockhamGenerator::hcKernWrite(transKernel, 12)
              << "outputA_I[iOffset + (lidy + loop*" << 16 / reShapeFactor
              << ")*" << params.fft_N[0]
              << " + lidx + starting_index_yx] = xy_s[index].y; }" << std::endl;
          break;
        case HCFFT_HERMITIAN_INTERLEAVED:
        case HCFFT_HERMITIAN_PLANAR:
          return HCFFT_INVALID;
        case HCFFT_REAL:
          break;
        default:
          return HCFFT_INVALID;
      }

      StockhamGenerator::hcKernWrite(transKernel, 6) << "}" << std::endl;  // end for
      StockhamGenerator::hcKernWrite(transKernel, 3) << "}" << std::endl;  // end else
    }
    StockhamGenerator::hcKernWrite(transKernel, 0) << "}).wait();\n}}\n" << std::endl;

    strKernel = transKernel.str();

    if (!params.fft_3StepTwiddle) break;
  }
  return HCFFT_SUCCEEDS;
}
}  // end of namespace hcfft_transpose_generator
