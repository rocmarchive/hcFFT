#include <iostream>
#include <cassert>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include "hcfftlib.h"

static bool Lfirst;

namespace StockhamGenerator {
// Precision
enum Precision {
  P_SINGLE,
  P_DOUBLE,
};

template <Precision PR>
inline size_t PrecisionWidth() {
  switch(PR) {
    case P_SINGLE:
      return 1;

    case P_DOUBLE:
      return 2;

    default:
      assert(false);
      return 1;
  }
}

inline std::string FloatToStr(double f) {
  std::stringstream ss;
  ss.precision(16);
  ss << std::scientific << f;
  return ss.str();
}

typedef std::pair<std::string, std::string> stringpair;
inline stringpair ComplexMul(const char* type, const char* a, const char* b, bool forward = true) {
  stringpair result;

  if((strcmp(type, "float_2")) != 0) {
    result.first = "(";
  }

  result.first += type;

  if((strcmp(type, "float_2")) != 0) {
    result.first += ")";
  }

  result.first += " ((";
  result.first += a;
  result.first += ".x * ";
  result.first += b;
  result.first += (forward ? ".x - " : ".x + ");
  result.first += a;
  result.first += ".y * ";
  result.first += b;
  result.first += ".y),";
  result.second = "(";
  result.second += a;
  result.second += ".y * ";
  result.second += b;
  result.second += (forward ? ".x + " : ".x - ");
  result.second += a;
  result.second += ".x * ";
  result.second += b;
  result.second += ".y))";
  return result;
}

// Register data base types
template <Precision PR>
inline std::string RegBaseType(size_t count) {
  switch(PR) {
    case P_SINGLE:
      switch(count) {
        case 1:
          return "float";

        case 2:
          return "float_2";

        case 4:
          return "float_4";

        default:
          assert(false);
          return "";
      }

      break;

    case P_DOUBLE:
      switch(count) {
        case 1:
          return "double";

        case 2:
          return "double_2";

        case 4:
          return "double_4";

        default:
          assert(false);
          return "";
      }

      break;

    default:
      assert(false);
      return "";
  }
}

template <Precision PR>
inline std::string FloatSuffix() {
  // Suffix for constants
  std::string sfx;

  switch(PR) {
    case P_SINGLE:
      sfx = "f";
      break;

    case P_DOUBLE:
      sfx = "";
      break;

    default:
      assert(false);
  }

  return sfx;
}

inline std::string ButterflyName(size_t radix, size_t count, bool fwd, const hcfftPlanHandle plHandle) {
  std::string str;

  if(fwd) {
    str += "Fwd";
  } else {
    str += "Inv";
  }

  str += "Rad";
  str += SztToStr(radix);
  str += "B";
  str += SztToStr(count);
  str += "H";
  str += SztToStr(plHandle);
  return str;
}

inline std::string PassName(const hcfftPlanHandle plHandle, size_t pos, bool fwd) {
  std::string str;

  if(fwd) {
    str += "Fwd";
  } else {
    str += "Inv";
  }

  str += "Pass";
  str += SztToStr(plHandle);
  str += SztToStr(pos);
  return str;
}

inline std::string TwTableName() {
  return "twiddles";
}

inline std::string TwTableLargeName() {
  return "twiddle_dee";
}

inline std::string TwTableLargeFunc() {
  return "TW3step";
}

// Twiddle factors table for large N
// used in 3-step algorithm
class TwiddleTableLarge {
  size_t N; // length
  size_t X, Y;
  size_t tableSize;
  double* wc, *ws; // cosine, sine arrays

 public:
  TwiddleTableLarge(size_t length) : N(length) {
    X = size_t(1) << ARBITRARY::TWIDDLE_DEE;
    Y = DivRoundingUp<size_t> (CeilPo2(N), ARBITRARY::TWIDDLE_DEE);
    tableSize = X * Y;
    // Allocate memory for the tables
    wc = new double[tableSize];
    ws = new double[tableSize];
  }

  ~TwiddleTableLarge() {
    // Free
    delete[] wc;
    delete[] ws;
  }

  template <Precision PR>
  void TwiddleLargeAV(std::string &twStr) {
    std::stringstream ss;
    const double TWO_PI = -6.283185307179586476925286766559;
    // Generate the table
    size_t nt = 0;
    double phi = TWO_PI / double (N);

    for (size_t iY = 0; iY < Y; ++iY) {
      size_t i = size_t(1) << (iY * ARBITRARY::TWIDDLE_DEE);

      for (size_t iX = 0; iX < X; ++iX) {
        size_t j = i * iX;
        double c = cos(phi * (double)j);
        double s = sin(phi * (double)j);
        //if (fabs(c) < 1.0E-12)  c = 0.0;
        //if (fabs(s) < 1.0E-12)  s = 0.0;
        wc[nt]   = c;
        ws[nt++] = s;
      }
    }

    std::string sfx = FloatSuffix<PR>();
    // Stringize the table
    nt = 0;
    ss << "\n\t";
    ss << RegBaseType<PR>(2);
    ss << " twiddlel ";
    ss << "[" << Y* X << "] = {\n";

    for (size_t iY = 0; iY < Y; ++iY) {
      for (size_t iX = 0; iX < X; ++iX) {
        char cv[64], sv[64];
        sprintf(cv, "%036.34lf", wc[nt]);
        sprintf(sv, "%036.34lf", ws[nt++]);
        ss << RegBaseType<PR>(2);
        ss << "(";
        ss << cv;
        ss << sfx;
        ss << ", ";
        ss << sv;
        ss << sfx;
        ss << ")";
        ss << ", ";
      }
    }

    ss << "};\n\n";
    // Construct array view from twiddlel array
    ss << RegBaseType<PR>(2);
    ss << " *";
    ss << TwTableLargeName();
    ss << " = hc::am_alloc(sizeof(";
    ss << RegBaseType<PR>(2);
    ss << ") * ";
    ss << SztToStr(Y * X);
    ss << ", acc, 0);\n\t";
    ss << "hc::am_copy(";
    ss << TwTableLargeName();
    ss << ", twiddlel, ";
    ss << SztToStr(Y * X);
    ss << " * sizeof(";
    ss << RegBaseType<PR>(2);
    ss << "));";
    twStr += ss.str();
  }

  template <Precision PR>
  void GenerateTwiddleTable(std::string &twStr) {
    if(!Lfirst) {
      std::stringstream ss;
      // Twiddle calc function
      ss << "inline ";
      ss << RegBaseType<PR>(2);
      ss << "\n" << TwTableLargeFunc() << "(unsigned int u, ";
      ss << RegBaseType<PR>(2);
      ss << " *";
      ss << TwTableLargeName();
      ss << ")\n{\n";
      ss << "\t" "unsigned int j = u & " << unsigned(X - 1) << ";\n";
      ss << "\t" ;
      ss << RegBaseType<PR>(2);
      ss << " result = ";
      ss << TwTableLargeName();
      ss << "[j];\n";

      for (size_t iY = 1; iY < Y; ++iY) {
        std::string phasor = TwTableLargeName();
        phasor += "[";
        phasor += SztToStr(iY * X) ;
        phasor += "+ j]";
        stringpair product = ComplexMul((RegBaseType<PR>(2)).c_str(), "result", phasor.c_str());
        ss << "\t" "u >>= " << unsigned (ARBITRARY::TWIDDLE_DEE) << ";\n";
        ss << "\t" "j = u & " << unsigned(X - 1) << ";\n";
        ss << "\t" "result = " << product.first << "\n";
        ss << "\t" "\t" << product.second << ";\n";
      }

      ss << "\t" "return result;\n}\n\n";
      twStr += ss.str();
      Lfirst = true;
    }
  }
};

// FFT butterfly
template <Precision PR>
class Butterfly {
  size_t radix;   // Base radix
  size_t count;       // Number of basic butterflies, valid values: 1,2,4
  bool fwd;     // FFT direction
  bool cReg;      // registers are complex numbers, .x (real), .y(imag)

  size_t BitReverse (size_t n, size_t N) const {
    return (N < 2) ? n : (BitReverse (n >> 1, N >> 1) | ((n & 1) != 0 ? (N >> 1) : 0));
  }
  void GenerateButterflyStr(std::string &bflyStr, const hcfftPlanHandle plHandle) const {
    std::string regType = cReg ? RegBaseType<PR>(2) : RegBaseType<PR>(count);
    // Function attribute
    bflyStr += "inline void \n";
    // Function name
    bflyStr += ButterflyName(radix, count, fwd, plHandle);
    // Function Arguments
    bflyStr += "(";

    for(size_t i = 0;; i++) {
      if(cReg) {
        bflyStr += regType;
        bflyStr += " *R";

        if(radix & (radix - 1)) {
          bflyStr += SztToStr(i);
        } else {
          bflyStr += SztToStr(BitReverse(i, radix));
        }
      } else {
        bflyStr += regType;
        bflyStr += " *R";
        bflyStr += SztToStr(i);
        bflyStr += ", ";  // real arguments
        bflyStr += regType;
        bflyStr += " *I";
        bflyStr += SztToStr(i);         // imaginary arguments
      }

      if(i == radix - 1) {
        bflyStr += ")";
        break;
      } else {
        bflyStr += ", ";
      }
    }

    bflyStr += " __attribute__((hc))\n{\n\n";

    // Temporary variables
    // Allocate temporary variables if we are not using complex registers (cReg = 0) or if cReg is true, then
    // allocate temporary variables only for non power-of-2 radices
    if( (radix & (radix - 1)) || (!cReg) ) {
      bflyStr += "\t";

      if(cReg) {
        bflyStr += RegBaseType<PR>(1);
      } else {
        bflyStr += regType;
      }

      for(size_t i = 0;; i++) {
        bflyStr += " TR";
        bflyStr += SztToStr(i);
        bflyStr += ","; // real arguments
        bflyStr += " TI";
        bflyStr += SztToStr(i);     // imaginary arguments

        if(i == radix - 1) {
          bflyStr += ";";
          break;
        } else {
          bflyStr += ",";
        }
      }
    } else {
      bflyStr += "\t";
      bflyStr += RegBaseType<PR>(2);
      bflyStr += " T;";
    }

    bflyStr += "\n\n\t";

    // Butterfly for different radices
    switch(radix) {
      case 2: {
          if(cReg) {
            bflyStr +=
              "(R1[0]) = (R0[0]) - (R1[0]);\n\t"
              "(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t";
          } else {
            bflyStr +=
              "TR0 = (R0[0]) + (R1[0]);\n\t"
              "TI0 = (I0[0]) + (I1[0]);\n\t"
              "TR1 = (R0[0]) - (R1[0]);\n\t"
              "TI1 = (I0[0]) - (I1[0]);\n\t";
          }
        }
        break;

      case 3: {
          if(fwd) {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R1[0]).x + (R2[0]).x;\n\t"
                "TR1 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) + C3QB*((R1[0]).y - (R2[0]).y);\n\t"
                "TR2 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) - C3QB*((R1[0]).y - (R2[0]).y);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R1[0]).y + (R2[0]).y;\n\t"
                "TI1 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) - C3QB*((R1[0]).x - (R2[0]).x);\n\t"
                "TI2 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) + C3QB*((R1[0]).x - (R2[0]).x);\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R1[0] + R2[0];\n\t"
                "TR1 = (R0[0] - C3QA*(R1[0] + R2[0])) + C3QB*(I1[0] - I2[0]);\n\t"
                "TR2 = (R0[0] - C3QA*(R1[0] + R2[0])) - C3QB*(I1[0] - I2[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I1[0] + I2[0];\n\t"
                "TI1 = (I0[0] - C3QA*(I1[0] + I2[0])) - C3QB*(R1[0] - R2[0]);\n\t"
                "TI2 = (I0[0] - C3QA*(I1[0] + I2[0])) + C3QB*(R1[0] - R2[0]);\n\t";
            }
          } else {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R1[0]).x + (R2[0]).x;\n\t"
                "TR1 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) - C3QB*((R1[0]).y - (R2[0]).y);\n\t"
                "TR2 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) + C3QB*((R1[0]).y - (R2[0]).y);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R1[0]).y + (R2[0]).y;\n\t"
                "TI1 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) + C3QB*((R1[0]).x - (R2[0]).x);\n\t"
                "TI2 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) - C3QB*((R1[0]).x - (R2[0]).x);\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R1[0] + R2[0];\n\t"
                "TR1 = (R0[0] - C3QA*(R1[0] + R2[0])) - C3QB*(I1[0] - I2[0]);\n\t"
                "TR2 = (R0[0] - C3QA*(R1[0] + R2[0])) + C3QB*(I1[0] - I2[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I1[0] + I2[0];\n\t"
                "TI1 = (I0[0] - C3QA*(I1[0] + I2[0])) + C3QB*(R1[0] - R2[0]);\n\t"
                "TI2 = (I0[0] - C3QA*(I1[0] + I2[0])) - C3QB*(R1[0] - R2[0]);\n\t";
            }
          }
        }
        break;

      case 4: {
          if(fwd) {
            if(cReg) {
              bflyStr +=
                "(R1[0]) = (R0[0]) - (R1[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t"
                "(R3[0]) = (R2[0]) - (R3[0]);\n\t"
                "(R2[0]) = 2.0f * (R2[0]) - (R3[0]);\n\t"
                "\n\t"
                "(R2[0]) = (R0[0]) - (R2[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R2[0]);\n\t"
                "(R3[0]) = (R1[0]) + fvect2(-(R3[0]).y, (R3[0]).x);\n\t"
                "(R1[0]) = 2.0f * (R1[0]) - (R3[0]);\n\t";
            } else {
              bflyStr +=
                "TR0 = (R0[0]) + (R2[0]) + (R1[0]) + (R3[0]);\n\t"
                "TR1 = (R0[0]) - (R2[0]) + (I1[0]) - (I3[0]);\n\t"
                "TR2 = (R0[0]) + (R2[0]) - (R1[0]) - (R3[0]);\n\t"
                "TR3 = (R0[0]) - (R2[0]) - (I1[0]) + (I3[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (I0[0]) + (I2[0]) + (I1[0]) + (I3[0]);\n\t"
                "TI1 = (I0[0]) - (I2[0]) - (R1[0]) + (R3[0]);\n\t"
                "TI2 = (I0[0]) + (I2[0]) - (I1[0]) - (I3[0]);\n\t"
                "TI3 = (I0[0]) - (I2[0]) + (R1[0]) - (R3[0]);\n\t";
            }
          } else {
            if(cReg) {
              bflyStr +=
                "(R1[0]) = (R0[0]) - (R1[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t"
                "(R3[0]) = (R2[0]) - (R3[0]);\n\t"
                "(R2[0]) = 2.0f * (R2[0]) - (R3[0]);\n\t"
                "\n\t"
                "(R2[0]) = (R0[0]) - (R2[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R2[0]);\n\t"
                "(R3[0]) = (R1[0]) + fvect2((R3[0]).y, -(R3[0]).x);\n\t"
                "(R1[0]) = 2.0f * (R1[0]) - (R3[0]);\n\t";
            } else {
              bflyStr +=
                "TR0 = (R0[0]) + (R2[0]) + (R1[0]) + (R3[0]);\n\t"
                "TR1 = (R0[0]) - (R2[0]) - (I1[0]) + (I3[0]);\n\t"
                "TR2 = (R0[0]) + (R2[0]) - (R1[0]) - (R3[0]);\n\t"
                "TR3 = (R0[0]) - (R2[0]) + (I1[0]) - (I3[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (I0[0]) + (I2[0]) + (I1[0]) + (I3[0]);\n\t"
                "TI1 = (I0[0]) - (I2[0]) + (R1[0]) - (R3[0]);\n\t"
                "TI2 = (I0[0]) + (I2[0]) - (I1[0]) - (I3[0]);\n\t"
                "TI3 = (I0[0]) - (I2[0]) - (R1[0]) + (R3[0]);\n\t";
            }
          }
        }
        break;

      case 5: {
          if(fwd) {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R1[0]).x + (R2[0]).x + (R3[0]).x + (R4[0]).x;\n\t"
                "TR1 = ((R0[0]).x - C5QC*((R2[0]).x + (R3[0]).x)) + C5QB*((R1[0]).y - (R4[0]).y) + C5QD*((R2[0]).y - (R3[0]).y) + C5QA*(((R1[0]).x - (R2[0]).x) + ((R4[0]).x - (R3[0]).x));\n\t"
                "TR4 = ((R0[0]).x - C5QC*((R2[0]).x + (R3[0]).x)) - C5QB*((R1[0]).y - (R4[0]).y) - C5QD*((R2[0]).y - (R3[0]).y) + C5QA*(((R1[0]).x - (R2[0]).x) + ((R4[0]).x - (R3[0]).x));\n\t"
                "TR2 = ((R0[0]).x - C5QC*((R1[0]).x + (R4[0]).x)) - C5QB*((R2[0]).y - (R3[0]).y) + C5QD*((R1[0]).y - (R4[0]).y) + C5QA*(((R2[0]).x - (R1[0]).x) + ((R3[0]).x - (R4[0]).x));\n\t"
                "TR3 = ((R0[0]).x - C5QC*((R1[0]).x + (R4[0]).x)) + C5QB*((R2[0]).y - (R3[0]).y) - C5QD*((R1[0]).y - (R4[0]).y) + C5QA*(((R2[0]).x - (R1[0]).x) + ((R3[0]).x - (R4[0]).x));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R1[0]).y + (R2[0]).y + (R3[0]).y + (R4[0]).y;\n\t"
                "TI1 = ((R0[0]).y - C5QC*((R2[0]).y + (R3[0]).y)) - C5QB*((R1[0]).x - (R4[0]).x) - C5QD*((R2[0]).x - (R3[0]).x) + C5QA*(((R1[0]).y - (R2[0]).y) + ((R4[0]).y - (R3[0]).y));\n\t"
                "TI4 = ((R0[0]).y - C5QC*((R2[0]).y + (R3[0]).y)) + C5QB*((R1[0]).x - (R4[0]).x) + C5QD*((R2[0]).x - (R3[0]).x) + C5QA*(((R1[0]).y - (R2[0]).y) + ((R4[0]).y - (R3[0]).y));\n\t"
                "TI2 = ((R0[0]).y - C5QC*((R1[0]).y + (R4[0]).y)) + C5QB*((R2[0]).x - (R3[0]).x) - C5QD*((R1[0]).x - (R4[0]).x) + C5QA*(((R2[0]).y - (R1[0]).y) + ((R3[0]).y - (R4[0]).y));\n\t"
                "TI3 = ((R0[0]).y - C5QC*((R1[0]).y + (R4[0]).y)) - C5QB*((R2[0]).x - (R3[0]).x) + C5QD*((R1[0]).x - (R4[0]).x) + C5QA*(((R2[0]).y - (R1[0]).y) + ((R3[0]).y - (R4[0]).y));\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R1[0] + R2[0] + R3[0] + R4[0];\n\t"
                "TR1 = (R0[0] - C5QC*(R2[0] + R3[0])) + C5QB*(I1[0] - I4[0]) + C5QD*(I2[0] - I3[0]) + C5QA*((R1[0] - R2[0]) + (R4[0] - R3[0]));\n\t"
                "TR4 = (R0[0] - C5QC*(R2[0] + R3[0])) - C5QB*(I1[0] - I4[0]) - C5QD*(I2[0] - I3[0]) + C5QA*((R1[0] - R2[0]) + (R4[0] - R3[0]));\n\t"
                "TR2 = (R0[0] - C5QC*(R1[0] + R4[0])) - C5QB*(I2[0] - I3[0]) + C5QD*(I1[0] - I4[0]) + C5QA*((R2[0] - R1[0]) + (R3[0] - R4[0]));\n\t"
                "TR3 = (R0[0] - C5QC*(R1[0] + R4[0])) + C5QB*(I2[0] - I3[0]) - C5QD*(I1[0] - I4[0]) + C5QA*((R2[0] - R1[0]) + (R3[0] - R4[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I1[0] + I2[0] + I3[0] + I4[0];\n\t"
                "TI1 = (I0[0] - C5QC*(I2[0] + I3[0])) - C5QB*(R1[0] - R4[0]) - C5QD*(R2[0] - R3[0]) + C5QA*((I1[0] - I2[0]) + (I4[0] - I3[0]));\n\t"
                "TI4 = (I0[0] - C5QC*(I2[0] + I3[0])) + C5QB*(R1[0] - R4[0]) + C5QD*(R2[0] - R3[0]) + C5QA*((I1[0] - I2[0]) + (I4[0] - I3[0]));\n\t"
                "TI2 = (I0[0] - C5QC*(I1[0] + I4[0])) + C5QB*(R2[0] - R3[0]) - C5QD*(R1[0] - R4[0]) + C5QA*((I2[0] - I1[0]) + (I3[0] - I4[0]));\n\t"
                "TI3 = (I0[0] - C5QC*(I1[0] + I4[0])) - C5QB*(R2[0] - R3[0]) + C5QD*(R1[0] - R4[0]) + C5QA*((I2[0] - I1[0]) + (I3[0] - I4[0]));\n\t";
            }
          } else {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R1[0]).x + (R2[0]).x + (R3[0]).x + (R4[0]).x;\n\t"
                "TR1 = ((R0[0]).x - C5QC*((R2[0]).x + (R3[0]).x)) - C5QB*((R1[0]).y - (R4[0]).y) - C5QD*((R2[0]).y - (R3[0]).y) + C5QA*(((R1[0]).x - (R2[0]).x) + ((R4[0]).x - (R3[0]).x));\n\t"
                "TR4 = ((R0[0]).x - C5QC*((R2[0]).x + (R3[0]).x)) + C5QB*((R1[0]).y - (R4[0]).y) + C5QD*((R2[0]).y - (R3[0]).y) + C5QA*(((R1[0]).x - (R2[0]).x) + ((R4[0]).x - (R3[0]).x));\n\t"
                "TR2 = ((R0[0]).x - C5QC*((R1[0]).x + (R4[0]).x)) + C5QB*((R2[0]).y - (R3[0]).y) - C5QD*((R1[0]).y - (R4[0]).y) + C5QA*(((R2[0]).x - (R1[0]).x) + ((R3[0]).x - (R4[0]).x));\n\t"
                "TR3 = ((R0[0]).x - C5QC*((R1[0]).x + (R4[0]).x)) - C5QB*((R2[0]).y - (R3[0]).y) + C5QD*((R1[0]).y - (R4[0]).y) + C5QA*(((R2[0]).x - (R1[0]).x) + ((R3[0]).x - (R4[0]).x));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R1[0]).y + (R2[0]).y + (R3[0]).y + (R4[0]).y;\n\t"
                "TI1 = ((R0[0]).y - C5QC*((R2[0]).y + (R3[0]).y)) + C5QB*((R1[0]).x - (R4[0]).x) + C5QD*((R2[0]).x - (R3[0]).x) + C5QA*(((R1[0]).y - (R2[0]).y) + ((R4[0]).y - (R3[0]).y));\n\t"
                "TI4 = ((R0[0]).y - C5QC*((R2[0]).y + (R3[0]).y)) - C5QB*((R1[0]).x - (R4[0]).x) - C5QD*((R2[0]).x - (R3[0]).x) + C5QA*(((R1[0]).y - (R2[0]).y) + ((R4[0]).y - (R3[0]).y));\n\t"
                "TI2 = ((R0[0]).y - C5QC*((R1[0]).y + (R4[0]).y)) - C5QB*((R2[0]).x - (R3[0]).x) + C5QD*((R1[0]).x - (R4[0]).x) + C5QA*(((R2[0]).y - (R1[0]).y) + ((R3[0]).y - (R4[0]).y));\n\t"
                "TI3 = ((R0[0]).y - C5QC*((R1[0]).y + (R4[0]).y)) + C5QB*((R2[0]).x - (R3[0]).x) - C5QD*((R1[0]).x - (R4[0]).x) + C5QA*(((R2[0]).y - (R1[0]).y) + ((R3[0]).y - (R4[0]).y));\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R1[0] + R2[0] + R3[0] + R4[0];\n\t"
                "TR1 = (R0[0] - C5QC*(R2[0] + R3[0])) - C5QB*(I1[0] - I4[0]) - C5QD*(I2[0] - I3[0]) + C5QA*((R1[0] - R2[0]) + (R4[0] - R3[0]));\n\t"
                "TR4 = (R0[0] - C5QC*(R2[0] + R3[0])) + C5QB*(I1[0] - I4[0]) + C5QD*(I2[0] - I3[0]) + C5QA*((R1[0] - R2[0]) + (R4[0] - R3[0]));\n\t"
                "TR2 = (R0[0] - C5QC*(R1[0] + R4[0])) + C5QB*(I2[0] - I3[0]) - C5QD*(I1[0] - I4[0]) + C5QA*((R2[0] - R1[0]) + (R3[0] - R4[0]));\n\t"
                "TR3 = (R0[0] - C5QC*(R1[0] + R4[0])) - C5QB*(I2[0] - I3[0]) + C5QD*(I1[0] - I4[0]) + C5QA*((R2[0] - R1[0]) + (R3[0] - R4[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I1[0] + I2[0] + I3[0] + I4[0];\n\t"
                "TI1 = (I0[0] - C5QC*(I2[0] + I3[0])) + C5QB*(R1[0] - R4[0]) + C5QD*(R2[0] - R3[0]) + C5QA*((I1[0] - I2[0]) + (I4[0] - I3[0]));\n\t"
                "TI4 = (I0[0] - C5QC*(I2[0] + I3[0])) - C5QB*(R1[0] - R4[0]) - C5QD*(R2[0] - R3[0]) + C5QA*((I1[0] - I2[0]) + (I4[0] - I3[0]));\n\t"
                "TI2 = (I0[0] - C5QC*(I1[0] + I4[0])) - C5QB*(R2[0] - R3[0]) + C5QD*(R1[0] - R4[0]) + C5QA*((I2[0] - I1[0]) + (I3[0] - I4[0]));\n\t"
                "TI3 = (I0[0] - C5QC*(I1[0] + I4[0])) + C5QB*(R2[0] - R3[0]) - C5QD*(R1[0] - R4[0]) + C5QA*((I2[0] - I1[0]) + (I3[0] - I4[0]));\n\t";
            }
          }
        }
        break;

      case 6: {
          if(fwd) {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R2[0]).x + (R4[0]).x;\n\t"
                "TR2 = ((R0[0]).x - C3QA*((R2[0]).x + (R4[0]).x)) + C3QB*((R2[0]).y - (R4[0]).y);\n\t"
                "TR4 = ((R0[0]).x - C3QA*((R2[0]).x + (R4[0]).x)) - C3QB*((R2[0]).y - (R4[0]).y);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R2[0]).y + (R4[0]).y;\n\t"
                "TI2 = ((R0[0]).y - C3QA*((R2[0]).y + (R4[0]).y)) - C3QB*((R2[0]).x - (R4[0]).x);\n\t"
                "TI4 = ((R0[0]).y - C3QA*((R2[0]).y + (R4[0]).y)) + C3QB*((R2[0]).x - (R4[0]).x);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = (R1[0]).x + (R3[0]).x + (R5[0]).x;\n\t"
                "TR3 = ((R1[0]).x - C3QA*((R3[0]).x + (R5[0]).x)) + C3QB*((R3[0]).y - (R5[0]).y);\n\t"
                "TR5 = ((R1[0]).x - C3QA*((R3[0]).x + (R5[0]).x)) - C3QB*((R3[0]).y - (R5[0]).y);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = (R1[0]).y + (R3[0]).y + (R5[0]).y;\n\t"
                "TI3 = ((R1[0]).y - C3QA*((R3[0]).y + (R5[0]).y)) - C3QB*((R3[0]).x - (R5[0]).x);\n\t"
                "TI5 = ((R1[0]).y - C3QA*((R3[0]).y + (R5[0]).y)) + C3QB*((R3[0]).x - (R5[0]).x);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).x = TR0 + TR1;\n\t"
                "(R1[0]).x = TR2 + ( C3QA*TR3 + C3QB*TI3);\n\t"
                "(R2[0]).x = TR4 + (-C3QA*TR5 + C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).y = TI0 + TI1;\n\t"
                "(R1[0]).y = TI2 + (-C3QB*TR3 + C3QA*TI3);\n\t"
                "(R2[0]).y = TI4 + (-C3QB*TR5 - C3QA*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R3[0]).x = TR0 - TR1;\n\t"
                "(R4[0]).x = TR2 - ( C3QA*TR3 + C3QB*TI3);\n\t"
                "(R5[0]).x = TR4 - (-C3QA*TR5 + C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R3[0]).y = TI0 - TI1;\n\t"
                "(R4[0]).y = TI2 - (-C3QB*TR3 + C3QA*TI3);\n\t"
                "(R5[0]).y = TI4 - (-C3QB*TR5 - C3QA*TI5);\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R2[0] + R4[0];\n\t"
                "TR2 = (R0[0] - C3QA*(R2[0] + R4[0])) + C3QB*(I2[0] - I4[0]);\n\t"
                "TR4 = (R0[0] - C3QA*(R2[0] + R4[0])) - C3QB*(I2[0] - I4[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I2[0] + I4[0];\n\t"
                "TI2 = (I0[0] - C3QA*(I2[0] + I4[0])) - C3QB*(R2[0] - R4[0]);\n\t"
                "TI4 = (I0[0] - C3QA*(I2[0] + I4[0])) + C3QB*(R2[0] - R4[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = R1[0] + R3[0] + R5[0];\n\t"
                "TR3 = (R1[0] - C3QA*(R3[0] + R5[0])) + C3QB*(I3[0] - I5[0]);\n\t"
                "TR5 = (R1[0] - C3QA*(R3[0] + R5[0])) - C3QB*(I3[0] - I5[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = I1[0] + I3[0] + I5[0];\n\t"
                "TI3 = (I1[0] - C3QA*(I3[0] + I5[0])) - C3QB*(R3[0] - R5[0]);\n\t"
                "TI5 = (I1[0] - C3QA*(I3[0] + I5[0])) + C3QB*(R3[0] - R5[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]) = TR0 + TR1;\n\t"
                "(R1[0]) = TR2 + ( C3QA*TR3 + C3QB*TI3);\n\t"
                "(R2[0]) = TR4 + (-C3QA*TR5 + C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I0[0]) = TI0 + TI1;\n\t"
                "(I1[0]) = TI2 + (-C3QB*TR3 + C3QA*TI3);\n\t"
                "(I2[0]) = TI4 + (-C3QB*TR5 - C3QA*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R3[0]) = TR0 - TR1;\n\t"
                "(R4[0]) = TR2 - ( C3QA*TR3 + C3QB*TI3);\n\t"
                "(R5[0]) = TR4 - (-C3QA*TR5 + C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I3[0]) = TI0 - TI1;\n\t"
                "(I4[0]) = TI2 - (-C3QB*TR3 + C3QA*TI3);\n\t"
                "(I5[0]) = TI4 - (-C3QB*TR5 - C3QA*TI5);\n\t";
            }
          } else {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R2[0]).x + (R4[0]).x;\n\t"
                "TR2 = ((R0[0]).x - C3QA*((R2[0]).x + (R4[0]).x)) - C3QB*((R2[0]).y - (R4[0]).y);\n\t"
                "TR4 = ((R0[0]).x - C3QA*((R2[0]).x + (R4[0]).x)) + C3QB*((R2[0]).y - (R4[0]).y);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R2[0]).y + (R4[0]).y;\n\t"
                "TI2 = ((R0[0]).y - C3QA*((R2[0]).y + (R4[0]).y)) + C3QB*((R2[0]).x - (R4[0]).x);\n\t"
                "TI4 = ((R0[0]).y - C3QA*((R2[0]).y + (R4[0]).y)) - C3QB*((R2[0]).x - (R4[0]).x);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = (R1[0]).x + (R3[0]).x + (R5[0]).x;\n\t"
                "TR3 = ((R1[0]).x - C3QA*((R3[0]).x + (R5[0]).x)) - C3QB*((R3[0]).y - (R5[0]).y);\n\t"
                "TR5 = ((R1[0]).x - C3QA*((R3[0]).x + (R5[0]).x)) + C3QB*((R3[0]).y - (R5[0]).y);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = (R1[0]).y + (R3[0]).y + (R5[0]).y;\n\t"
                "TI3 = ((R1[0]).y - C3QA*((R3[0]).y + (R5[0]).y)) + C3QB*((R3[0]).x - (R5[0]).x);\n\t"
                "TI5 = ((R1[0]).y - C3QA*((R3[0]).y + (R5[0]).y)) - C3QB*((R3[0]).x - (R5[0]).x);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).x = TR0 + TR1;\n\t"
                "(R1[0]).x = TR2 + ( C3QA*TR3 - C3QB*TI3);\n\t"
                "(R2[0]).x = TR4 + (-C3QA*TR5 - C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).y = TI0 + TI1;\n\t"
                "(R1[0]).y = TI2 + ( C3QB*TR3 + C3QA*TI3);\n\t"
                "(R2[0]).y = TI4 + ( C3QB*TR5 - C3QA*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R3[0]).x = TR0 - TR1;\n\t"
                "(R4[0]).x = TR2 - ( C3QA*TR3 - C3QB*TI3);\n\t"
                "(R5[0]).x = TR4 - (-C3QA*TR5 - C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R3[0]).y = TI0 - TI1;\n\t"
                "(R4[0]).y = TI2 - ( C3QB*TR3 + C3QA*TI3);\n\t"
                "(R5[0]).y = TI4 - ( C3QB*TR5 - C3QA*TI5);\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R2[0] + R4[0];\n\t"
                "TR2 = (R0[0] - C3QA*(R2[0] + R4[0])) - C3QB*(I2[0] - I4[0]);\n\t"
                "TR4 = (R0[0] - C3QA*(R2[0] + R4[0])) + C3QB*(I2[0] - I4[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I2[0] + I4[0];\n\t"
                "TI2 = (I0[0] - C3QA*(I2[0] + I4[0])) + C3QB*(R2[0] - R4[0]);\n\t"
                "TI4 = (I0[0] - C3QA*(I2[0] + I4[0])) - C3QB*(R2[0] - R4[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = R1[0] + R3[0] + R5[0];\n\t"
                "TR3 = (R1[0] - C3QA*(R3[0] + R5[0])) - C3QB*(I3[0] - I5[0]);\n\t"
                "TR5 = (R1[0] - C3QA*(R3[0] + R5[0])) + C3QB*(I3[0] - I5[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = I1[0] + I3[0] + I5[0];\n\t"
                "TI3 = (I1[0] - C3QA*(I3[0] + I5[0])) + C3QB*(R3[0] - R5[0]);\n\t"
                "TI5 = (I1[0] - C3QA*(I3[0] + I5[0])) - C3QB*(R3[0] - R5[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]) = TR0 + TR1;\n\t"
                "(R1[0]) = TR2 + ( C3QA*TR3 - C3QB*TI3);\n\t"
                "(R2[0]) = TR4 + (-C3QA*TR5 - C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I0[0]) = TI0 + TI1;\n\t"
                "(I1[0]) = TI2 + ( C3QB*TR3 + C3QA*TI3);\n\t"
                "(I2[0]) = TI4 + ( C3QB*TR5 - C3QA*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R3[0]) = TR0 - TR1;\n\t"
                "(R4[0]) = TR2 - ( C3QA*TR3 - C3QB*TI3);\n\t"
                "(R5[0]) = TR4 - (-C3QA*TR5 - C3QB*TI5);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I3[0]) = TI0 - TI1;\n\t"
                "(I4[0]) = TI2 - ( C3QB*TR3 + C3QA*TI3);\n\t"
                "(I5[0]) = TI4 - ( C3QB*TR5 - C3QA*TI5);\n\t";
            }
          }
        }
        break;

      case 7: {
          static const char* C7SFR = "\
				/*FFT7 Forward Real */ \n\
				\n\
				pr0 = R1[0] + R6[0]; \n\
				pi0 = I1[0] + I6[0]; \n\
				pr1 = R1[0] - R6[0]; \n\
				pi1 = I1[0] - I6[0]; \n\
				pr2 = R2[0] + R5[0]; \n\
				pi2 = I2[0] + I5[0]; \n\
				pr3 = R2[0] - R5[0]; \n\
				pi3 = I2[0] - I5[0]; \n\
				pr4 = R4[0] + R3[0]; \n\
				pi4 = I4[0] + I3[0]; \n\
				pr5 = R4[0] - R3[0]; \n\
				pi5 = I4[0] - I3[0]; \n\
				\n\
				pr6 = pr2 + pr0; \n\
				pi6 = pi2 + pi0; \n\
				qr4 = pr2 - pr0; \n\
				qi4 = pi2 - pi0; \n\
				qr2 = pr0 - pr4; \n\
				qi2 = pi0 - pi4; \n\
				qr3 = pr4 - pr2; \n\
				qi3 = pi4 - pi2; \n\
				pr7 = pr5 + pr3; \n\
				pi7 = pi5 + pi3; \n\
				qr7 = pr5 - pr3; \n\
				qi7 = pi5 - pi3; \n\
				qr6 = pr1 - pr5; \n\
				qi6 = pi1 - pi5; \n\
				qr8 = pr3 - pr1; \n\
				qi8 = pi3 - pi1; \n\
				qr1 = pr6 + pr4; \n\
				qi1 = pi6 + pi4; \n\
				qr5 = pr7 + pr1; \n\
				qi5 = pi7 + pi1; \n\
				qr0 = R0[0] + qr1; \n\
				qi0 = I0[0] + qi1; \n\
				\n\
				qr1 *= C7Q1; \n\
				qi1 *= C7Q1; \n\
				qr2 *= C7Q2; \n\
				qi2 *= C7Q2; \n\
				qr3 *= C7Q3; \n\
				qi3 *= C7Q3; \n\
				qr4 *= C7Q4; \n\
				qi4 *= C7Q4; \n\
				\n\
				qr5 *= (C7Q5); \n\
				qi5 *= (C7Q5); \n\
				qr6 *= (C7Q6); \n\
				qi6 *= (C7Q6); \n\
				qr7 *= (C7Q7); \n\
				qi7 *= (C7Q7); \n\
				qr8 *= (C7Q8); \n\
				qi8 *= (C7Q8); \n\
				\n\
				pr0 =  qr0 + qr1; \n\
				pi0 =  qi0 + qi1; \n\
				pr1 =  qr2 + qr3; \n\
				pi1 =  qi2 + qi3; \n\
				pr2 =  qr4 - qr3; \n\
				pi2 =  qi4 - qi3; \n\
				pr3 = -qr2 - qr4; \n\
				pi3 = -qi2 - qi4; \n\
				pr4 =  qr6 + qr7; \n\
				pi4 =  qi6 + qi7; \n\
				pr5 =  qr8 - qr7; \n\
				pi5 =  qi8 - qi7; \n\
				pr6 = -qr8 - qr6; \n\
				pi6 = -qi8 - qi6; \n\
				pr7 =  pr0 + pr1; \n\
				pi7 =  pi0 + pi1; \n\
				pr8 =  pr0 + pr2; \n\
				pi8 =  pi0 + pi2; \n\
				pr9 =  pr0 + pr3; \n\
				pi9 =  pi0 + pi3; \n\
				qr6 =  pr4 + qr5; \n\
				qi6 =  pi4 + qi5; \n\
				qr7 =  pr5 + qr5; \n\
				qi7 =  pi5 + qi5; \n\
				qr8 =  pr6 + qr5; \n\
				qi8 =  pi6 + qi5; \n\
				\n\
				TR0 = qr0; TI0 = qi0; \n\
				TR1 = pr7 + qi6; \n\
				TI1 = pi7 - qr6; \n\
				TR2 = pr9 + qi8; \n\
				TI2 = pi9 - qr8; \n\
				TR3 = pr8 - qi7; \n\
				TI3 = pi8 + qr7; \n\
				TR4 = pr8 + qi7; \n\
				TI4 = pi8 - qr7; \n\
				TR5 = pr9 - qi8; \n\
				TI5 = pi9 + qr8; \n\
				TR6 = pr7 - qi6; \n\
				TI6 = pi7 + qr6; \n\
				";
          static const char* C7SBR = "\
				/*FFT7 Backward Real */ \n\
				\n\
				pr0 = R1[0] + R6[0]; \n\
				pi0 = I1[0] + I6[0]; \n\
				pr1 = R1[0] - R6[0]; \n\
				pi1 = I1[0] - I6[0]; \n\
				pr2 = R2[0] + R5[0]; \n\
				pi2 = I2[0] + I5[0]; \n\
				pr3 = R2[0] - R5[0]; \n\
				pi3 = I2[0] - I5[0]; \n\
				pr4 = R4[0] + R3[0]; \n\
				pi4 = I4[0] + I3[0]; \n\
				pr5 = R4[0] - R3[0]; \n\
				pi5 = I4[0] - I3[0]; \n\
				\n\
				pr6 = pr2 + pr0; \n\
				pi6 = pi2 + pi0; \n\
				qr4 = pr2 - pr0; \n\
				qi4 = pi2 - pi0; \n\
				qr2 = pr0 - pr4; \n\
				qi2 = pi0 - pi4; \n\
				qr3 = pr4 - pr2; \n\
				qi3 = pi4 - pi2; \n\
				pr7 = pr5 + pr3; \n\
				pi7 = pi5 + pi3; \n\
				qr7 = pr5 - pr3; \n\
				qi7 = pi5 - pi3; \n\
				qr6 = pr1 - pr5; \n\
				qi6 = pi1 - pi5; \n\
				qr8 = pr3 - pr1; \n\
				qi8 = pi3 - pi1; \n\
				qr1 = pr6 + pr4; \n\
				qi1 = pi6 + pi4; \n\
				qr5 = pr7 + pr1; \n\
				qi5 = pi7 + pi1; \n\
				qr0 = R0[0] + qr1; \n\
				qi0 = I0[0] + qi1; \n\
				\n\
				qr1 *= C7Q1; \n\
				qi1 *= C7Q1; \n\
				qr2 *= C7Q2; \n\
				qi2 *= C7Q2; \n\
				qr3 *= C7Q3; \n\
				qi3 *= C7Q3; \n\
				qr4 *= C7Q4; \n\
				qi4 *= C7Q4; \n\
				\n\
				qr5 *= -(C7Q5); \n\
				qi5 *= -(C7Q5); \n\
				qr6 *= -(C7Q6); \n\
				qi6 *= -(C7Q6); \n\
				qr7 *= -(C7Q7); \n\
				qi7 *= -(C7Q7); \n\
				qr8 *= -(C7Q8); \n\
				qi8 *= -(C7Q8); \n\
				\n\
				pr0 =  qr0 + qr1; \n\
				pi0 =  qi0 + qi1; \n\
				pr1 =  qr2 + qr3; \n\
				pi1 =  qi2 + qi3; \n\
				pr2 =  qr4 - qr3; \n\
				pi2 =  qi4 - qi3; \n\
				pr3 = -qr2 - qr4; \n\
				pi3 = -qi2 - qi4; \n\
				pr4 =  qr6 + qr7; \n\
				pi4 =  qi6 + qi7; \n\
				pr5 =  qr8 - qr7; \n\
				pi5 =  qi8 - qi7; \n\
				pr6 = -qr8 - qr6; \n\
				pi6 = -qi8 - qi6; \n\
				pr7 =  pr0 + pr1; \n\
				pi7 =  pi0 + pi1; \n\
				pr8 =  pr0 + pr2; \n\
				pi8 =  pi0 + pi2; \n\
				pr9 =  pr0 + pr3; \n\
				pi9 =  pi0 + pi3; \n\
				qr6 =  pr4 + qr5; \n\
				qi6 =  pi4 + qi5; \n\
				qr7 =  pr5 + qr5; \n\
				qi7 =  pi5 + qi5; \n\
				qr8 =  pr6 + qr5; \n\
				qi8 =  pi6 + qi5; \n\
			\n\
				TR0 = qr0; TI0 = qi0; \n\
				TR1 = pr7 + qi6; \n\
				TI1 = pi7 - qr6; \n\
				TR2 = pr9 + qi8; \n\
				TI2 = pi9 - qr8; \n\
				TR3 = pr8 - qi7; \n\
				TI3 = pi8 + qr7; \n\
				TR4 = pr8 + qi7; \n\
				TI4 = pi8 - qr7; \n\
				TR5 = pr9 - qi8; \n\
				TI5 = pi9 + qr8; \n\
				TR6 = pr7 - qi6; \n\
				TI6 = pi7 + qr6; \n\
			";
          static const char* C7SFC = "\
			/*FFT7 Forward Complex */ \n\
			\n\
				p0 = R1[0] + R6[0]; \n\
				p1 = R1[0] - R6[0]; \n\
				p2 = R2[0] + R5[0]; \n\
				p3 = R2[0] - R5[0]; \n\
				p4 = R4[0] + R3[0]; \n\
				p5 = R4[0] - R3[0]; \n\
			\n\
				p6 = p2 + p0; \n\
				q4 = p2 - p0; \n\
				q2 = p0 - p4; \n\
				q3 = p4 - p2; \n\
				p7 = p5 + p3; \n\
				q7 = p5 - p3; \n\
				q6 = p1 - p5; \n\
				q8 = p3 - p1; \n\
				q1 = p6 + p4; \n\
				q5 = p7 + p1; \n\
				q0 = R0[0] + q1; \n\
			\n\
				q1 *= C7Q1; \n\
				q2 *= C7Q2; \n\
				q3 *= C7Q3; \n\
				q4 *= C7Q4; \n\
			\n\
				q5 *= (C7Q5); \n\
				q6 *= (C7Q6); \n\
				q7 *= (C7Q7); \n\
				q8 *= (C7Q8); \n\
			\n\
				p0 = q0 + q1; \n\
				p1 = q2 + q3; \n\
				p2 = q4 - q3; \n\
				p3 = -q2 - q4; \n\
				p4 = q6 + q7; \n\
				p5 = q8 - q7; \n\
				p6 = -q8 - q6; \n\
				p7 = p0 + p1; \n\
				p8 = p0 + p2; \n\
				p9 = p0 + p3; \n\
				q6 = p4 + q5; \n\
				q7 = p5 + q5; \n\
				q8 = p6 + q5; \n\
			\n\
				R0[0] = q0; \n\
				(R1[0]).x = p7.x + q6.y; \n\
				(R1[0]).y = p7.y - q6.x; \n\
				(R2[0]).x = p9.x + q8.y; \n\
				(R2[0]).y = p9.y - q8.x; \n\
				(R3[0]).x = p8.x - q7.y; \n\
				(R3[0]).y = p8.y + q7.x; \n\
				(R4[0]).x = p8.x + q7.y; \n\
				(R4[0]).y = p8.y - q7.x; \n\
				(R5[0]).x = p9.x - q8.y; \n\
				(R5[0]).y = p9.y + q8.x; \n\
				(R6[0]).x = p7.x - q6.y; \n\
				(R6[0]).y = p7.y + q6.x; \n\
			";
          static const char* C7SBC = "\
			/*FFT7 Backward Complex */ \n\
			\n\
				p0 = R1[0] + R6[0]; \n\
				p1 = R1[0] - R6[0]; \n\
				p2 = R2[0] + R5[0]; \n\
				p3 = R2[0] - R5[0]; \n\
				p4 = R4[0] + R3[0]; \n\
				p5 = R4[0] - R3[0]; \n\
			\n\
				p6 = p2 + p0; \n\
				q4 = p2 - p0; \n\
				q2 = p0 - p4; \n\
				q3 = p4 - p2; \n\
				p7 = p5 + p3; \n\
				q7 = p5 - p3; \n\
				q6 = p1 - p5; \n\
				q8 = p3 - p1; \n\
				q1 = p6 + p4; \n\
				q5 = p7 + p1; \n\
				q0 = R0[0] + q1; \n\
			\n\
				q1 *= C7Q1; \n\
				q2 *= C7Q2; \n\
				q3 *= C7Q3; \n\
				q4 *= C7Q4; \n\
			\n\
				q5 *= -(C7Q5); \n\
				q6 *= -(C7Q6); \n\
				q7 *= -(C7Q7); \n\
				q8 *= -(C7Q8); \n\
			\n\
				p0 = q0 + q1; \n\
				p1 = q2 + q3; \n\
				p2 = q4 - q3; \n\
				p3 = -q2 - q4; \n\
				p4 = q6 + q7; \n\
				p5 = q8 - q7; \n\
				p6 = -q8 - q6; \n\
				p7 = p0 + p1; \n\
				p8 = p0 + p2; \n\
				p9 = p0 + p3; \n\
				q6 = p4 + q5; \n\
				q7 = p5 + q5; \n\
				q8 = p6 + q5; \n\
			\n\
				R0[0] = q0; \n\
				(R1[0]).x = p7.x + q6.y; \n\
				(R1[0]).y = p7.y - q6.x; \n\
				(R2[0]).x = p9.x + q8.y; \n\
				(R2[0]).y = p9.y - q8.x; \n\
				(R3[0]).x = p8.x - q7.y; \n\
				(R3[0]).y = p8.y + q7.x; \n\
				(R4[0]).x = p8.x + q7.y; \n\
				(R4[0]).y = p8.y - q7.x; \n\
				(R5[0]).x = p9.x - q8.y; \n\
				(R5[0]).y = p9.y + q8.x; \n\
				(R6[0]).x = p7.x - q6.y; \n\
				(R6[0]).y = p7.y + q6.x; \n\
			";

          if (!cReg) {
            for (size_t i = 0; i < 10; i++) {
              bflyStr += regType + " pr" + SztToStr(i) + ", pi" + SztToStr(i) + ";\n\t";
            }

            for (size_t i = 0; i < 9; i++) {
              bflyStr += regType + " qr" + SztToStr(i) + ", qi" + SztToStr(i) + ";\n\t";
            }

            if (fwd) {
              bflyStr += C7SFR;
            } else {
              bflyStr += C7SBR;
            }
          } else {
            for (size_t i = 0; i < 10; i++) {
              bflyStr += regType + " p" + SztToStr(i) + ";\n\t";
            }

            for (size_t i = 0; i < 9; i++) {
              bflyStr += regType + " q" + SztToStr(i) + ";\n\t";
            }

            if (fwd) {
              bflyStr += C7SFC;
            } else {
              bflyStr += C7SBC;
            }
          }
        }
        break;

      case 8: {
          if(fwd) {
            if(cReg) {
              bflyStr +=
                "(R1[0]) = (R0[0]) - (R1[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t"
                "(R3[0]) = (R2[0]) - (R3[0]);\n\t"
                "(R2[0]) = 2.0f * (R2[0]) - (R3[0]);\n\t"
                "(R5[0]) = (R4[0]) - (R5[0]);\n\t"
                "(R4[0]) = 2.0f * (R4[0]) - (R5[0]);\n\t"
                "(R7[0]) = (R6[0]) - (R7[0]);\n\t"
                "(R6[0]) = 2.0f * (R6[0]) - (R7[0]);\n\t"
                "\n\t"
                "(R2[0]) = (R0[0]) - (R2[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R2[0]);\n\t"
                "(R3[0]) = (R1[0]) + fvect2(-(R3[0]).y, (R3[0]).x);\n\t"
                "(R1[0]) = 2.0f * (R1[0]) - (R3[0]);\n\t"
                "(R6[0]) = (R4[0]) - (R6[0]);\n\t"
                "(R4[0]) = 2.0f * (R4[0]) - (R6[0]);\n\t"
                "(R7[0]) = (R5[0]) + fvect2(-(R7[0]).y, (R7[0]).x);\n\t"
                "(R5[0]) = 2.0f * (R5[0]) - (R7[0]);\n\t"
                "\n\t"
                "(R4[0]) = (R0[0]) - (R4[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R4[0]);\n\t"
                "(R5[0]) = ((R1[0]) - C8Q * (R5[0])) - C8Q * fvect2((R5[0]).y, -(R5[0]).x);\n\t"
                "(R1[0]) = 2.0f * (R1[0]) - (R5[0]);\n\t"
                "(R6[0]) = (R2[0]) + fvect2(-(R6[0]).y, (R6[0]).x);\n\t"
                "(R2[0]) = 2.0f * (R2[0]) - (R6[0]);\n\t"
                "(R7[0]) = ((R3[0]) + C8Q * (R7[0])) - C8Q * fvect2((R7[0]).y, -(R7[0]).x);\n\t"
                "(R3[0]) = 2.0f * (R3[0]) - (R7[0]);\n\t";
            } else {
              bflyStr +=
                "TR0 = (R0[0]) + (R4[0]) + (R2[0]) + (R6[0]) +     (R1[0])             +     (R3[0])             +     (R5[0])             +     (R7[0])            ;\n\t"
                "TR1 = (R0[0]) - (R4[0]) + (I2[0]) - (I6[0]) + C8Q*(R1[0]) + C8Q*(I1[0]) - C8Q*(R3[0]) + C8Q*(I3[0]) - C8Q*(R5[0]) - C8Q*(I5[0]) + C8Q*(R7[0]) - C8Q*(I7[0]);\n\t"
                "TR2 = (R0[0]) + (R4[0]) - (R2[0]) - (R6[0])             +     (I1[0])             -     (I3[0])             +     (I5[0])             -     (I7[0]);\n\t"
                "TR3 = (R0[0]) - (R4[0]) - (I2[0]) + (I6[0]) - C8Q*(R1[0]) + C8Q*(I1[0]) + C8Q*(R3[0]) + C8Q*(I3[0]) + C8Q*(R5[0]) - C8Q*(I5[0]) - C8Q*(R7[0]) - C8Q*(I7[0]);\n\t"
                "TR4 = (R0[0]) + (R4[0]) + (R2[0]) + (R6[0]) -     (R1[0])             -     (R3[0])             -     (R5[0])             -     (R7[0])            ;\n\t"
                "TR5 = (R0[0]) - (R4[0]) + (I2[0]) - (I6[0]) - C8Q*(R1[0]) - C8Q*(I1[0]) + C8Q*(R3[0]) - C8Q*(I3[0]) + C8Q*(R5[0]) + C8Q*(I5[0]) - C8Q*(R7[0]) + C8Q*(I7[0]);\n\t"
                "TR6 = (R0[0]) + (R4[0]) - (R2[0]) - (R6[0])             -    (I1[0])              +     (I3[0])             -     (I5[0])             +     (I7[0]);\n\t"
                "TR7 = (R0[0]) - (R4[0]) - (I2[0]) + (I6[0]) + C8Q*(R1[0]) - C8Q*(I1[0]) - C8Q*(R3[0]) - C8Q*(I3[0]) - C8Q*(R5[0]) + C8Q*(I5[0]) + C8Q*(R7[0]) + C8Q*(I7[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (I0[0]) + (I4[0]) + (I2[0]) + (I6[0])             +     (I1[0])             +     (I3[0])             +     (I5[0])             +     (I7[0]);\n\t"
                "TI1 = (I0[0]) - (I4[0]) - (R2[0]) + (R6[0]) - C8Q*(R1[0]) + C8Q*(I1[0]) - C8Q*(R3[0]) - C8Q*(I3[0]) + C8Q*(R5[0]) - C8Q*(I5[0]) + C8Q*(R7[0]) + C8Q*(I7[0]);\n\t"
                "TI2 = (I0[0]) + (I4[0]) - (I2[0]) - (I6[0]) -     (R1[0])             +     (R3[0])             -     (R5[0])             +     (R7[0])            ;\n\t"
                "TI3 = (I0[0]) - (I4[0]) + (R2[0]) - (R6[0]) - C8Q*(R1[0]) - C8Q*(I1[0]) - C8Q*(R3[0]) + C8Q*(I3[0]) + C8Q*(R5[0]) + C8Q*(I5[0]) + C8Q*(R7[0]) - C8Q*(I7[0]);\n\t"
                "TI4 = (I0[0]) + (I4[0]) + (I2[0]) + (I6[0])             -    (I1[0])              -     (I3[0])             -     (I5[0])             -     (I7[0]);\n\t"
                "TI5 = (I0[0]) - (I4[0]) - (R2[0]) + (R6[0]) + C8Q*(R1[0]) - C8Q*(I1[0]) + C8Q*(R3[0]) + C8Q*(I3[0]) - C8Q*(R5[0]) + C8Q*(I5[0]) - C8Q*(R7[0]) - C8Q*(I7[0]);\n\t"
                "TI6 = (I0[0]) + (I4[0]) - (I2[0]) - (I6[0]) +     (R1[0])             -     (R3[0])             +     (R5[0])             -     (R7[0])            ;\n\t"
                "TI7 = (I0[0]) - (I4[0]) + (R2[0]) - (R6[0]) + C8Q*(R1[0]) + C8Q*(I1[0]) + C8Q*(R3[0]) - C8Q*(I3[0]) - C8Q*(R5[0]) - C8Q*(I5[0]) - C8Q*(R7[0]) + C8Q*(I7[0]);\n\t";
            }
          } else {
            if(cReg) {
              bflyStr +=
                "(R1[0]) = (R0[0]) - (R1[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t"
                "(R3[0]) = (R2[0]) - (R3[0]);\n\t"
                "(R2[0]) = 2.0f * (R2[0]) - (R3[0]);\n\t"
                "(R5[0]) = (R4[0]) - (R5[0]);\n\t"
                "(R4[0]) = 2.0f * (R4[0]) - (R5[0]);\n\t"
                "(R7[0]) = (R6[0]) - (R7[0]);\n\t"
                "(R6[0]) = 2.0f * (R6[0]) - (R7[0]);\n\t"
                "\n\t"
                "(R2[0]) = (R0[0]) - (R2[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R2[0]);\n\t"
                "(R3[0]) = (R1[0]) + fvect2((R3[0]).y, -(R3[0]).x);\n\t"
                "(R1[0]) = 2.0f * (R1[0]) - (R3[0]);\n\t"
                "(R6[0]) = (R4[0]) - (R6[0]);\n\t"
                "(R4[0]) = 2.0f * (R4[0]) - (R6[0]);\n\t"
                "(R7[0]) = (R5[0]) + fvect2((R7[0]).y, -(R7[0]).x);\n\t"
                "(R5[0]) = 2.0f * (R5[0]) - (R7[0]);\n\t"
                "\n\t"
                "(R4[0]) = (R0[0]) - (R4[0]);\n\t"
                "(R0[0]) = 2.0f * (R0[0]) - (R4[0]);\n\t"
                "(R5[0]) = ((R1[0]) - C8Q * (R5[0])) + C8Q * fvect2((R5[0]).y, -(R5[0]).x);\n\t"
                "(R1[0]) = 2.0f * (R1[0]) - (R5[0]);\n\t"
                "(R6[0]) = (R2[0]) + fvect2((R6[0]).y, -(R6[0]).x);\n\t"
                "(R2[0]) = 2.0f * (R2[0]) - (R6[0]);\n\t"
                "(R7[0]) = ((R3[0]) + C8Q * (R7[0])) + C8Q * fvect2((R7[0]).y, -(R7[0]).x);\n\t"
                "(R3[0]) = 2.0f * (R3[0]) - (R7[0]);\n\t";
            } else {
              bflyStr +=
                "TR0 = (R0[0]) + (R4[0]) + (R2[0]) + (R6[0]) +     (R1[0])             +     (R3[0])             +     (R5[0])             +     (R7[0])            ;\n\t"
                "TR1 = (R0[0]) - (R4[0]) - (I2[0]) + (I6[0]) + C8Q*(R1[0]) - C8Q*(I1[0]) - C8Q*(R3[0]) - C8Q*(I3[0]) - C8Q*(R5[0]) + C8Q*(I5[0]) + C8Q*(R7[0]) + C8Q*(I7[0]);\n\t"
                "TR2 = (R0[0]) + (R4[0]) - (R2[0]) - (R6[0])             -     (I1[0])             +     (I3[0])             -     (I5[0])             +     (I7[0]);\n\t"
                "TR3 = (R0[0]) - (R4[0]) + (I2[0]) - (I6[0]) - C8Q*(R1[0]) - C8Q*(I1[0]) + C8Q*(R3[0]) - C8Q*(I3[0]) + C8Q*(R5[0]) + C8Q*(I5[0]) - C8Q*(R7[0]) + C8Q*(I7[0]);\n\t"
                "TR4 = (R0[0]) + (R4[0]) + (R2[0]) + (R6[0]) -     (R1[0])             -    (R3[0])              -     (R5[0])             -     (R7[0])            ;\n\t"
                "TR5 = (R0[0]) - (R4[0]) - (I2[0]) + (I6[0]) - C8Q*(R1[0]) + C8Q*(I1[0]) + C8Q*(R3[0]) + C8Q*(I3[0]) + C8Q*(R5[0]) - C8Q*(I5[0]) - C8Q*(R7[0]) - C8Q*(I7[0]);\n\t"
                "TR6 = (R0[0]) + (R4[0]) - (R2[0]) - (R6[0])             +     (I1[0])             -     (I3[0])             +     (I5[0])             -     (I7[0]);\n\t"
                "TR7 = (R0[0]) - (R4[0]) + (I2[0]) - (I6[0]) + C8Q*(R1[0]) + C8Q*(I1[0]) - C8Q*(R3[0]) + C8Q*(I3[0]) - C8Q*(R5[0]) - C8Q*(I5[0]) + C8Q*(R7[0]) - C8Q*(I7[0]);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (I0[0]) + (I4[0]) + (I2[0]) + (I6[0])             +     (I1[0])             +    (I3[0])              +     (I5[0])             +     (I7[0]);\n\t"
                "TI1 = (I0[0]) - (I4[0]) + (R2[0]) - (R6[0]) + C8Q*(R1[0]) + C8Q*(I1[0]) + C8Q*(R3[0]) - C8Q*(I3[0]) - C8Q*(R5[0]) - C8Q*(I5[0]) - C8Q*(R7[0]) + C8Q*(I7[0]);\n\t"
                "TI2 = (I0[0]) + (I4[0]) - (I2[0]) - (I6[0]) +     (R1[0])             -     (R3[0])             +     (R5[0])             -     (R7[0])            ;\n\t"
                "TI3 = (I0[0]) - (I4[0]) - (R2[0]) + (R6[0]) + C8Q*(R1[0]) - C8Q*(I1[0]) + C8Q*(R3[0]) + C8Q*(I3[0]) - C8Q*(R5[0]) + C8Q*(I5[0]) - C8Q*(R7[0]) - C8Q*(I7[0]);\n\t"
                "TI4 = (I0[0]) + (I4[0]) + (I2[0]) + (I6[0])             -     (I1[0])             -     (I3[0])             -     (I5[0])             -     (I7[0]);\n\t"
                "TI5 = (I0[0]) - (I4[0]) + (R2[0]) - (R6[0]) - C8Q*(R1[0]) - C8Q*(I1[0]) - C8Q*(R3[0]) + C8Q*(I3[0]) + C8Q*(R5[0]) + C8Q*(I5[0]) + C8Q*(R7[0]) - C8Q*(I7[0]);\n\t"
                "TI6 = (I0[0]) + (I4[0]) - (I2[0]) - (I6[0]) -     (R1[0])             +     (R3[0])             -     (R5[0])             +     (R7[0])            ;\n\t"
                "TI7 = (I0[0]) - (I4[0]) - (R2[0]) + (R6[0]) - C8Q*(R1[0]) + C8Q*(I1[0]) - C8Q*(R3[0]) - C8Q*(I3[0]) + C8Q*(R5[0]) - C8Q*(I5[0]) + C8Q*(R7[0]) + C8Q*(I7[0]);\n\t";
            }
          }
        }
        break;

      case 10: {
          if(fwd) {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R2[0]).x + (R4[0]).x + (R6[0]).x + (R8[0]).x;\n\t"
                "TR2 = ((R0[0]).x - C5QC*((R4[0]).x + (R6[0]).x)) + C5QB*((R2[0]).y - (R8[0]).y) + C5QD*((R4[0]).y - (R6[0]).y) + C5QA*(((R2[0]).x - (R4[0]).x) + ((R8[0]).x - (R6[0]).x));\n\t"
                "TR8 = ((R0[0]).x - C5QC*((R4[0]).x + (R6[0]).x)) - C5QB*((R2[0]).y - (R8[0]).y) - C5QD*((R4[0]).y - (R6[0]).y) + C5QA*(((R2[0]).x - (R4[0]).x) + ((R8[0]).x - (R6[0]).x));\n\t"
                "TR4 = ((R0[0]).x - C5QC*((R2[0]).x + (R8[0]).x)) - C5QB*((R4[0]).y - (R6[0]).y) + C5QD*((R2[0]).y - (R8[0]).y) + C5QA*(((R4[0]).x - (R2[0]).x) + ((R6[0]).x - (R8[0]).x));\n\t"
                "TR6 = ((R0[0]).x - C5QC*((R2[0]).x + (R8[0]).x)) + C5QB*((R4[0]).y - (R6[0]).y) - C5QD*((R2[0]).y - (R8[0]).y) + C5QA*(((R4[0]).x - (R2[0]).x) + ((R6[0]).x - (R8[0]).x));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R2[0]).y + (R4[0]).y + (R6[0]).y + (R8[0]).y;\n\t"
                "TI2 = ((R0[0]).y - C5QC*((R4[0]).y + (R6[0]).y)) - C5QB*((R2[0]).x - (R8[0]).x) - C5QD*((R4[0]).x - (R6[0]).x) + C5QA*(((R2[0]).y - (R4[0]).y) + ((R8[0]).y - (R6[0]).y));\n\t"
                "TI8 = ((R0[0]).y - C5QC*((R4[0]).y + (R6[0]).y)) + C5QB*((R2[0]).x - (R8[0]).x) + C5QD*((R4[0]).x - (R6[0]).x) + C5QA*(((R2[0]).y - (R4[0]).y) + ((R8[0]).y - (R6[0]).y));\n\t"
                "TI4 = ((R0[0]).y - C5QC*((R2[0]).y + (R8[0]).y)) + C5QB*((R4[0]).x - (R6[0]).x) - C5QD*((R2[0]).x - (R8[0]).x) + C5QA*(((R4[0]).y - (R2[0]).y) + ((R6[0]).y - (R8[0]).y));\n\t"
                "TI6 = ((R0[0]).y - C5QC*((R2[0]).y + (R8[0]).y)) - C5QB*((R4[0]).x - (R6[0]).x) + C5QD*((R2[0]).x - (R8[0]).x) + C5QA*(((R4[0]).y - (R2[0]).y) + ((R6[0]).y - (R8[0]).y));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = (R1[0]).x + (R3[0]).x + (R5[0]).x + (R7[0]).x + (R9[0]).x;\n\t"
                "TR3 = ((R1[0]).x - C5QC*((R5[0]).x + (R7[0]).x)) + C5QB*((R3[0]).y - (R9[0]).y) + C5QD*((R5[0]).y - (R7[0]).y) + C5QA*(((R3[0]).x - (R5[0]).x) + ((R9[0]).x - (R7[0]).x));\n\t"
                "TR9 = ((R1[0]).x - C5QC*((R5[0]).x + (R7[0]).x)) - C5QB*((R3[0]).y - (R9[0]).y) - C5QD*((R5[0]).y - (R7[0]).y) + C5QA*(((R3[0]).x - (R5[0]).x) + ((R9[0]).x - (R7[0]).x));\n\t"
                "TR5 = ((R1[0]).x - C5QC*((R3[0]).x + (R9[0]).x)) - C5QB*((R5[0]).y - (R7[0]).y) + C5QD*((R3[0]).y - (R9[0]).y) + C5QA*(((R5[0]).x - (R3[0]).x) + ((R7[0]).x - (R9[0]).x));\n\t"
                "TR7 = ((R1[0]).x - C5QC*((R3[0]).x + (R9[0]).x)) + C5QB*((R5[0]).y - (R7[0]).y) - C5QD*((R3[0]).y - (R9[0]).y) + C5QA*(((R5[0]).x - (R3[0]).x) + ((R7[0]).x - (R9[0]).x));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = (R1[0]).y + (R3[0]).y + (R5[0]).y + (R7[0]).y + (R9[0]).y;\n\t"
                "TI3 = ((R1[0]).y - C5QC*((R5[0]).y + (R7[0]).y)) - C5QB*((R3[0]).x - (R9[0]).x) - C5QD*((R5[0]).x - (R7[0]).x) + C5QA*(((R3[0]).y - (R5[0]).y) + ((R9[0]).y - (R7[0]).y));\n\t"
                "TI9 = ((R1[0]).y - C5QC*((R5[0]).y + (R7[0]).y)) + C5QB*((R3[0]).x - (R9[0]).x) + C5QD*((R5[0]).x - (R7[0]).x) + C5QA*(((R3[0]).y - (R5[0]).y) + ((R9[0]).y - (R7[0]).y));\n\t"
                "TI5 = ((R1[0]).y - C5QC*((R3[0]).y + (R9[0]).y)) + C5QB*((R5[0]).x - (R7[0]).x) - C5QD*((R3[0]).x - (R9[0]).x) + C5QA*(((R5[0]).y - (R3[0]).y) + ((R7[0]).y - (R9[0]).y));\n\t"
                "TI7 = ((R1[0]).y - C5QC*((R3[0]).y + (R9[0]).y)) - C5QB*((R5[0]).x - (R7[0]).x) + C5QD*((R3[0]).x - (R9[0]).x) + C5QA*(((R5[0]).y - (R3[0]).y) + ((R7[0]).y - (R9[0]).y));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).x = TR0 + TR1;\n\t"
                "(R1[0]).x = TR2 + ( C5QE*TR3 + C5QD*TI3);\n\t"
                "(R2[0]).x = TR4 + ( C5QA*TR5 + C5QB*TI5);\n\t"
                "(R3[0]).x = TR6 + (-C5QA*TR7 + C5QB*TI7);\n\t"
                "(R4[0]).x = TR8 + (-C5QE*TR9 + C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).y = TI0 + TI1;\n\t"
                "(R1[0]).y = TI2 + (-C5QD*TR3 + C5QE*TI3);\n\t"
                "(R2[0]).y = TI4 + (-C5QB*TR5 + C5QA*TI5);\n\t"
                "(R3[0]).y = TI6 + (-C5QB*TR7 - C5QA*TI7);\n\t"
                "(R4[0]).y = TI8 + (-C5QD*TR9 - C5QE*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R5[0]).x = TR0 - TR1;\n\t"
                "(R6[0]).x = TR2 - ( C5QE*TR3 + C5QD*TI3);\n\t"
                "(R7[0]).x = TR4 - ( C5QA*TR5 + C5QB*TI5);\n\t"
                "(R8[0]).x = TR6 - (-C5QA*TR7 + C5QB*TI7);\n\t"
                "(R9[0]).x = TR8 - (-C5QE*TR9 + C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R5[0]).y = TI0 - TI1;\n\t"
                "(R6[0]).y = TI2 - (-C5QD*TR3 + C5QE*TI3);\n\t"
                "(R7[0]).y = TI4 - (-C5QB*TR5 + C5QA*TI5);\n\t"
                "(R8[0]).y = TI6 - (-C5QB*TR7 - C5QA*TI7);\n\t"
                "(R9[0]).y = TI8 - (-C5QD*TR9 - C5QE*TI9);\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R2[0] + R4[0] + R6[0] + R8[0];\n\t"
                "TR2 = (R0[0] - C5QC*(R4[0] + R6[0])) + C5QB*(I2[0] - I8[0]) + C5QD*(I4[0] - I6[0]) + C5QA*((R2[0] - R4[0]) + (R8[0] - R6[0]));\n\t"
                "TR8 = (R0[0] - C5QC*(R4[0] + R6[0])) - C5QB*(I2[0] - I8[0]) - C5QD*(I4[0] - I6[0]) + C5QA*((R2[0] - R4[0]) + (R8[0] - R6[0]));\n\t"
                "TR4 = (R0[0] - C5QC*(R2[0] + R8[0])) - C5QB*(I4[0] - I6[0]) + C5QD*(I2[0] - I8[0]) + C5QA*((R4[0] - R2[0]) + (R6[0] - R8[0]));\n\t"
                "TR6 = (R0[0] - C5QC*(R2[0] + R8[0])) + C5QB*(I4[0] - I6[0]) - C5QD*(I2[0] - I8[0]) + C5QA*((R4[0] - R2[0]) + (R6[0] - R8[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I2[0] + I4[0] + I6[0] + I8[0];\n\t"
                "TI2 = (I0[0] - C5QC*(I4[0] + I6[0])) - C5QB*(R2[0] - R8[0]) - C5QD*(R4[0] - R6[0]) + C5QA*((I2[0] - I4[0]) + (I8[0] - I6[0]));\n\t"
                "TI8 = (I0[0] - C5QC*(I4[0] + I6[0])) + C5QB*(R2[0] - R8[0]) + C5QD*(R4[0] - R6[0]) + C5QA*((I2[0] - I4[0]) + (I8[0] - I6[0]));\n\t"
                "TI4 = (I0[0] - C5QC*(I2[0] + I8[0])) + C5QB*(R4[0] - R6[0]) - C5QD*(R2[0] - R8[0]) + C5QA*((I4[0] - I2[0]) + (I6[0] - I8[0]));\n\t"
                "TI6 = (I0[0] - C5QC*(I2[0] + I8[0])) - C5QB*(R4[0] - R6[0]) + C5QD*(R2[0] - R8[0]) + C5QA*((I4[0] - I2[0]) + (I6[0] - I8[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = R1[0] + R3[0] + R5[0] + R7[0] + R9[0];\n\t"
                "TR3 = (R1[0] - C5QC*(R5[0] + R7[0])) + C5QB*(I3[0] - I9[0]) + C5QD*(I5[0] - I7[0]) + C5QA*((R3[0] - R5[0]) + (R9[0] - R7[0]));\n\t"
                "TR9 = (R1[0] - C5QC*(R5[0] + R7[0])) - C5QB*(I3[0] - I9[0]) - C5QD*(I5[0] - I7[0]) + C5QA*((R3[0] - R5[0]) + (R9[0] - R7[0]));\n\t"
                "TR5 = (R1[0] - C5QC*(R3[0] + R9[0])) - C5QB*(I5[0] - I7[0]) + C5QD*(I3[0] - I9[0]) + C5QA*((R5[0] - R3[0]) + (R7[0] - R9[0]));\n\t"
                "TR7 = (R1[0] - C5QC*(R3[0] + R9[0])) + C5QB*(I5[0] - I7[0]) - C5QD*(I3[0] - I9[0]) + C5QA*((R5[0] - R3[0]) + (R7[0] - R9[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = I1[0] + I3[0] + I5[0] + I7[0] + I9[0];\n\t"
                "TI3 = (I1[0] - C5QC*(I5[0] + I7[0])) - C5QB*(R3[0] - R9[0]) - C5QD*(R5[0] - R7[0]) + C5QA*((I3[0] - I5[0]) + (I9[0] - I7[0]));\n\t"
                "TI9 = (I1[0] - C5QC*(I5[0] + I7[0])) + C5QB*(R3[0] - R9[0]) + C5QD*(R5[0] - R7[0]) + C5QA*((I3[0] - I5[0]) + (I9[0] - I7[0]));\n\t"
                "TI5 = (I1[0] - C5QC*(I3[0] + I9[0])) + C5QB*(R5[0] - R7[0]) - C5QD*(R3[0] - R9[0]) + C5QA*((I5[0] - I3[0]) + (I7[0] - I9[0]));\n\t"
                "TI7 = (I1[0] - C5QC*(I3[0] + I9[0])) - C5QB*(R5[0] - R7[0]) + C5QD*(R3[0] - R9[0]) + C5QA*((I5[0] - I3[0]) + (I7[0] - I9[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]) = TR0 + TR1;\n\t"
                "(R1[0]) = TR2 + ( C5QE*TR3 + C5QD*TI3);\n\t"
                "(R2[0]) = TR4 + ( C5QA*TR5 + C5QB*TI5);\n\t"
                "(R3[0]) = TR6 + (-C5QA*TR7 + C5QB*TI7);\n\t"
                "(R4[0]) = TR8 + (-C5QE*TR9 + C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I0[0]) = TI0 + TI1;\n\t"
                "(I1[0]) = TI2 + (-C5QD*TR3 + C5QE*TI3);\n\t"
                "(I2[0]) = TI4 + (-C5QB*TR5 + C5QA*TI5);\n\t"
                "(I3[0]) = TI6 + (-C5QB*TR7 - C5QA*TI7);\n\t"
                "(I4[0]) = TI8 + (-C5QD*TR9 - C5QE*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R5[0]) = TR0 - TR1;\n\t"
                "(R6[0]) = TR2 - ( C5QE*TR3 + C5QD*TI3);\n\t"
                "(R7[0]) = TR4 - ( C5QA*TR5 + C5QB*TI5);\n\t"
                "(R8[0]) = TR6 - (-C5QA*TR7 + C5QB*TI7);\n\t"
                "(R9[0]) = TR8 - (-C5QE*TR9 + C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I5[0]) = TI0 - TI1;\n\t"
                "(I6[0]) = TI2 - (-C5QD*TR3 + C5QE*TI3);\n\t"
                "(I7[0]) = TI4 - (-C5QB*TR5 + C5QA*TI5);\n\t"
                "(I8[0]) = TI6 - (-C5QB*TR7 - C5QA*TI7);\n\t"
                "(I9[0]) = TI8 - (-C5QD*TR9 - C5QE*TI9);\n\t";
            }
          } else {
            if(cReg) {
              bflyStr +=
                "TR0 = (R0[0]).x + (R2[0]).x + (R4[0]).x + (R6[0]).x + (R8[0]).x;\n\t"
                "TR2 = ((R0[0]).x - C5QC*((R4[0]).x + (R6[0]).x)) - C5QB*((R2[0]).y - (R8[0]).y) - C5QD*((R4[0]).y - (R6[0]).y) + C5QA*(((R2[0]).x - (R4[0]).x) + ((R8[0]).x - (R6[0]).x));\n\t"
                "TR8 = ((R0[0]).x - C5QC*((R4[0]).x + (R6[0]).x)) + C5QB*((R2[0]).y - (R8[0]).y) + C5QD*((R4[0]).y - (R6[0]).y) + C5QA*(((R2[0]).x - (R4[0]).x) + ((R8[0]).x - (R6[0]).x));\n\t"
                "TR4 = ((R0[0]).x - C5QC*((R2[0]).x + (R8[0]).x)) + C5QB*((R4[0]).y - (R6[0]).y) - C5QD*((R2[0]).y - (R8[0]).y) + C5QA*(((R4[0]).x - (R2[0]).x) + ((R6[0]).x - (R8[0]).x));\n\t"
                "TR6 = ((R0[0]).x - C5QC*((R2[0]).x + (R8[0]).x)) - C5QB*((R4[0]).y - (R6[0]).y) + C5QD*((R2[0]).y - (R8[0]).y) + C5QA*(((R4[0]).x - (R2[0]).x) + ((R6[0]).x - (R8[0]).x));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = (R0[0]).y + (R2[0]).y + (R4[0]).y + (R6[0]).y + (R8[0]).y;\n\t"
                "TI2 = ((R0[0]).y - C5QC*((R4[0]).y + (R6[0]).y)) + C5QB*((R2[0]).x - (R8[0]).x) + C5QD*((R4[0]).x - (R6[0]).x) + C5QA*(((R2[0]).y - (R4[0]).y) + ((R8[0]).y - (R6[0]).y));\n\t"
                "TI8 = ((R0[0]).y - C5QC*((R4[0]).y + (R6[0]).y)) - C5QB*((R2[0]).x - (R8[0]).x) - C5QD*((R4[0]).x - (R6[0]).x) + C5QA*(((R2[0]).y - (R4[0]).y) + ((R8[0]).y - (R6[0]).y));\n\t"
                "TI4 = ((R0[0]).y - C5QC*((R2[0]).y + (R8[0]).y)) - C5QB*((R4[0]).x - (R6[0]).x) + C5QD*((R2[0]).x - (R8[0]).x) + C5QA*(((R4[0]).y - (R2[0]).y) + ((R6[0]).y - (R8[0]).y));\n\t"
                "TI6 = ((R0[0]).y - C5QC*((R2[0]).y + (R8[0]).y)) + C5QB*((R4[0]).x - (R6[0]).x) - C5QD*((R2[0]).x - (R8[0]).x) + C5QA*(((R4[0]).y - (R2[0]).y) + ((R6[0]).y - (R8[0]).y));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = (R1[0]).x + (R3[0]).x + (R5[0]).x + (R7[0]).x + (R9[0]).x;\n\t"
                "TR3 = ((R1[0]).x - C5QC*((R5[0]).x + (R7[0]).x)) - C5QB*((R3[0]).y - (R9[0]).y) - C5QD*((R5[0]).y - (R7[0]).y) + C5QA*(((R3[0]).x - (R5[0]).x) + ((R9[0]).x - (R7[0]).x));\n\t"
                "TR9 = ((R1[0]).x - C5QC*((R5[0]).x + (R7[0]).x)) + C5QB*((R3[0]).y - (R9[0]).y) + C5QD*((R5[0]).y - (R7[0]).y) + C5QA*(((R3[0]).x - (R5[0]).x) + ((R9[0]).x - (R7[0]).x));\n\t"
                "TR5 = ((R1[0]).x - C5QC*((R3[0]).x + (R9[0]).x)) + C5QB*((R5[0]).y - (R7[0]).y) - C5QD*((R3[0]).y - (R9[0]).y) + C5QA*(((R5[0]).x - (R3[0]).x) + ((R7[0]).x - (R9[0]).x));\n\t"
                "TR7 = ((R1[0]).x - C5QC*((R3[0]).x + (R9[0]).x)) - C5QB*((R5[0]).y - (R7[0]).y) + C5QD*((R3[0]).y - (R9[0]).y) + C5QA*(((R5[0]).x - (R3[0]).x) + ((R7[0]).x - (R9[0]).x));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = (R1[0]).y + (R3[0]).y + (R5[0]).y + (R7[0]).y + (R9[0]).y;\n\t"
                "TI3 = ((R1[0]).y - C5QC*((R5[0]).y + (R7[0]).y)) + C5QB*((R3[0]).x - (R9[0]).x) + C5QD*((R5[0]).x - (R7[0]).x) + C5QA*(((R3[0]).y - (R5[0]).y) + ((R9[0]).y - (R7[0]).y));\n\t"
                "TI9 = ((R1[0]).y - C5QC*((R5[0]).y + (R7[0]).y)) - C5QB*((R3[0]).x - (R9[0]).x) - C5QD*((R5[0]).x - (R7[0]).x) + C5QA*(((R3[0]).y - (R5[0]).y) + ((R9[0]).y - (R7[0]).y));\n\t"
                "TI5 = ((R1[0]).y - C5QC*((R3[0]).y + (R9[0]).y)) - C5QB*((R5[0]).x - (R7[0]).x) + C5QD*((R3[0]).x - (R9[0]).x) + C5QA*(((R5[0]).y - (R3[0]).y) + ((R7[0]).y - (R9[0]).y));\n\t"
                "TI7 = ((R1[0]).y - C5QC*((R3[0]).y + (R9[0]).y)) + C5QB*((R5[0]).x - (R7[0]).x) - C5QD*((R3[0]).x - (R9[0]).x) + C5QA*(((R5[0]).y - (R3[0]).y) + ((R7[0]).y - (R9[0]).y));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).x = TR0 + TR1;\n\t"
                "(R1[0]).x = TR2 + ( C5QE*TR3 - C5QD*TI3);\n\t"
                "(R2[0]).x = TR4 + ( C5QA*TR5 - C5QB*TI5);\n\t"
                "(R3[0]).x = TR6 + (-C5QA*TR7 - C5QB*TI7);\n\t"
                "(R4[0]).x = TR8 + (-C5QE*TR9 - C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]).y = TI0 + TI1;\n\t"
                "(R1[0]).y = TI2 + ( C5QD*TR3 + C5QE*TI3);\n\t"
                "(R2[0]).y = TI4 + ( C5QB*TR5 + C5QA*TI5);\n\t"
                "(R3[0]).y = TI6 + ( C5QB*TR7 - C5QA*TI7);\n\t"
                "(R4[0]).y = TI8 + ( C5QD*TR9 - C5QE*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R5[0]).x = TR0 - TR1;\n\t"
                "(R6[0]).x = TR2 - ( C5QE*TR3 - C5QD*TI3);\n\t"
                "(R7[0]).x = TR4 - ( C5QA*TR5 - C5QB*TI5);\n\t"
                "(R8[0]).x = TR6 - (-C5QA*TR7 - C5QB*TI7);\n\t"
                "(R9[0]).x = TR8 - (-C5QE*TR9 - C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R5[0]).y = TI0 - TI1;\n\t"
                "(R6[0]).y = TI2 - ( C5QD*TR3 + C5QE*TI3);\n\t"
                "(R7[0]).y = TI4 - ( C5QB*TR5 + C5QA*TI5);\n\t"
                "(R8[0]).y = TI6 - ( C5QB*TR7 - C5QA*TI7);\n\t"
                "(R9[0]).y = TI8 - ( C5QD*TR9 - C5QE*TI9);\n\t";
            } else {
              bflyStr +=
                "TR0 = R0[0] + R2[0] + R4[0] + R6[0] + R8[0];\n\t"
                "TR2 = (R0[0] - C5QC*(R4[0] + R6[0])) - C5QB*(I2[0] - I8[0]) - C5QD*(I4[0] - I6[0]) + C5QA*((R2[0] - R4[0]) + (R8[0] - R6[0]));\n\t"
                "TR8 = (R0[0] - C5QC*(R4[0] + R6[0])) + C5QB*(I2[0] - I8[0]) + C5QD*(I4[0] - I6[0]) + C5QA*((R2[0] - R4[0]) + (R8[0] - R6[0]));\n\t"
                "TR4 = (R0[0] - C5QC*(R2[0] + R8[0])) + C5QB*(I4[0] - I6[0]) - C5QD*(I2[0] - I8[0]) + C5QA*((R4[0] - R2[0]) + (R6[0] - R8[0]));\n\t"
                "TR6 = (R0[0] - C5QC*(R2[0] + R8[0])) - C5QB*(I4[0] - I6[0]) + C5QD*(I2[0] - I8[0]) + C5QA*((R4[0] - R2[0]) + (R6[0] - R8[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI0 = I0[0] + I2[0] + I4[0] + I6[0] + I8[0];\n\t"
                "TI2 = (I0[0] - C5QC*(I4[0] + I6[0])) + C5QB*(R2[0] - R8[0]) + C5QD*(R4[0] - R6[0]) + C5QA*((I2[0] - I4[0]) + (I8[0] - I6[0]));\n\t"
                "TI8 = (I0[0] - C5QC*(I4[0] + I6[0])) - C5QB*(R2[0] - R8[0]) - C5QD*(R4[0] - R6[0]) + C5QA*((I2[0] - I4[0]) + (I8[0] - I6[0]));\n\t"
                "TI4 = (I0[0] - C5QC*(I2[0] + I8[0])) - C5QB*(R4[0] - R6[0]) + C5QD*(R2[0] - R8[0]) + C5QA*((I4[0] - I2[0]) + (I6[0] - I8[0]));\n\t"
                "TI6 = (I0[0] - C5QC*(I2[0] + I8[0])) + C5QB*(R4[0] - R6[0]) - C5QD*(R2[0] - R8[0]) + C5QA*((I4[0] - I2[0]) + (I6[0] - I8[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TR1 = R1[0] + R3[0] + R5[0] + R7[0] + R9[0];\n\t"
                "TR3 = (R1[0] - C5QC*(R5[0] + R7[0])) - C5QB*(I3[0] - I9[0]) - C5QD*(I5[0] - I7[0]) + C5QA*((R3[0] - R5[0]) + (R9[0] - R7[0]));\n\t"
                "TR9 = (R1[0] - C5QC*(R5[0] + R7[0])) + C5QB*(I3[0] - I9[0]) + C5QD*(I5[0] - I7[0]) + C5QA*((R3[0] - R5[0]) + (R9[0] - R7[0]));\n\t"
                "TR5 = (R1[0] - C5QC*(R3[0] + R9[0])) + C5QB*(I5[0] - I7[0]) - C5QD*(I3[0] - I9[0]) + C5QA*((R5[0] - R3[0]) + (R7[0] - R9[0]));\n\t"
                "TR7 = (R1[0] - C5QC*(R3[0] + R9[0])) - C5QB*(I5[0] - I7[0]) + C5QD*(I3[0] - I9[0]) + C5QA*((R5[0] - R3[0]) + (R7[0] - R9[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "TI1 = I1[0] + I3[0] + I5[0] + I7[0] + I9[0];\n\t"
                "TI3 = (I1[0] - C5QC*(I5[0] + I7[0])) + C5QB*(R3[0] - R9[0]) + C5QD*(R5[0] - R7[0]) + C5QA*((I3[0] - I5[0]) + (I9[0] - I7[0]));\n\t"
                "TI9 = (I1[0] - C5QC*(I5[0] + I7[0])) - C5QB*(R3[0] - R9[0]) - C5QD*(R5[0] - R7[0]) + C5QA*((I3[0] - I5[0]) + (I9[0] - I7[0]));\n\t"
                "TI5 = (I1[0] - C5QC*(I3[0] + I9[0])) - C5QB*(R5[0] - R7[0]) + C5QD*(R3[0] - R9[0]) + C5QA*((I5[0] - I3[0]) + (I7[0] - I9[0]));\n\t"
                "TI7 = (I1[0] - C5QC*(I3[0] + I9[0])) + C5QB*(R5[0] - R7[0]) - C5QD*(R3[0] - R9[0]) + C5QA*((I5[0] - I3[0]) + (I7[0] - I9[0]));\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R0[0]) = TR0 + TR1;\n\t"
                "(R1[0]) = TR2 + ( C5QE*TR3 - C5QD*TI3);\n\t"
                "(R2[0]) = TR4 + ( C5QA*TR5 - C5QB*TI5);\n\t"
                "(R3[0]) = TR6 + (-C5QA*TR7 - C5QB*TI7);\n\t"
                "(R4[0]) = TR8 + (-C5QE*TR9 - C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I0[0]) = TI0 + TI1;\n\t"
                "(I1[0]) = TI2 + ( C5QD*TR3 + C5QE*TI3);\n\t"
                "(I2[0]) = TI4 + ( C5QB*TR5 + C5QA*TI5);\n\t"
                "(I3[0]) = TI6 + ( C5QB*TR7 - C5QA*TI7);\n\t"
                "(I4[0]) = TI8 + ( C5QD*TR9 - C5QE*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(R5[0]) = TR0 - TR1;\n\t"
                "(R6[0]) = TR2 - ( C5QE*TR3 - C5QD*TI3);\n\t"
                "(R7[0]) = TR4 - ( C5QA*TR5 - C5QB*TI5);\n\t"
                "(R8[0]) = TR6 - (-C5QA*TR7 - C5QB*TI7);\n\t"
                "(R9[0]) = TR8 - (-C5QE*TR9 - C5QD*TI9);\n\t";
              bflyStr += "\n\t";
              bflyStr +=
                "(I5[0]) = TI0 - TI1;\n\t"
                "(I6[0]) = TI2 - ( C5QD*TR3 + C5QE*TI3);\n\t"
                "(I7[0]) = TI4 - ( C5QB*TR5 + C5QA*TI5);\n\t"
                "(I8[0]) = TI6 - ( C5QB*TR7 - C5QA*TI7);\n\t"
                "(I9[0]) = TI8 - ( C5QD*TR9 - C5QE*TI9);\n\t";
            }
          }
        }
        break;

      default:
        assert(false);
    }

    bflyStr += "\n\t";

    // Assign results
    if( (radix & (radix - 1)) || (!cReg) ) {
      if( (radix != 10) && (radix != 6) ) {
        for(size_t i = 0; i < radix; i++) {
          if(cReg) {
            if (radix != 7) {
              bflyStr += "((R";
              bflyStr += SztToStr(i);
              bflyStr += "[0]).x) = TR";
              bflyStr += SztToStr(i);
              bflyStr += "; ";
              bflyStr += "((R";
              bflyStr += SztToStr(i);
              bflyStr += "[0]).y) = TI";
              bflyStr += SztToStr(i);
              bflyStr += ";\n\t";
            }
          } else {
            bflyStr += "(R";
            bflyStr += SztToStr(i);
            bflyStr += "[0]) = TR";
            bflyStr += SztToStr(i);
            bflyStr += "; ";
            bflyStr += "(I";
            bflyStr += SztToStr(i);
            bflyStr += "[0]) = TI";
            bflyStr += SztToStr(i);
            bflyStr += ";\n\t";
          }
        }
      }
    } else {
      for(size_t i = 0; i < radix; i++) {
        size_t j = BitReverse(i, radix);

        if(i < j) {
          bflyStr += "T = (R";
          bflyStr += SztToStr(i);
          bflyStr += "[0]); (R";
          bflyStr += SztToStr(i);
          bflyStr += "[0]) = (R";
          bflyStr += SztToStr(j);
          bflyStr += "[0]); (R";
          bflyStr += SztToStr(j);
          bflyStr += "[0]) = T;\n\t";
        }
      }
    }

    bflyStr += "\n}\n";
  }
 public:
  Butterfly(size_t radixVal, size_t countVal, bool fwdVal, bool cRegVal) : radix(radixVal), count(countVal), fwd(fwdVal), cReg(cRegVal) {}

  void GenerateButterfly(std::string &bflyStr, const hcfftPlanHandle plHandle) const {
    assert(count <= 4);

    if(count > 0) {
      GenerateButterflyStr(bflyStr, plHandle);
    }
  }
};
}
