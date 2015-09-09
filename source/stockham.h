#include <iostream>
#include <cassert>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include "hcfftlib.h"

using namespace std;

namespace StockhamGenerator
{
	// Precision
	enum Precision
	{
		P_SINGLE,
		P_DOUBLE,
	};

	template <Precision PR>
	inline size_t PrecisionWidth()
	{
		switch(PR)
		{
		case P_SINGLE:	return 1;
		case P_DOUBLE:	return 2;
		default:		assert(false); return 1;
		}
	}

	// Convert unsigned integers to string
	inline std::string SztToStr(size_t i)
	{
		std::stringstream ss;
		ss << i;
		return ss.str();
	}

	inline std::string FloatToStr(double f)
	{
		std::stringstream ss;
		ss.precision(16);
		ss << std::scientific << f;
		return ss.str();
	}

        typedef std::pair<std::string,std::string> stringpair;
	inline stringpair ComplexMul(const char *type, const char * a, const char * b, bool forward = true)
	{
		stringpair result;
		result.first = "(";
		result.first += type;
		result.first += ") ((";
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
	inline std::string RegBaseType(size_t count)
	{
		switch(PR)
		{
		case P_SINGLE:
			switch(count)
			{
			case 1: return "float";
			case 2: return "float_2";
			case 4: return "float_4";
			default: assert(false); return "";
			}
			break;
		case P_DOUBLE:
			switch(count)
			{
			case 1: return "double";
			case 2: return "double_2";
			case 4: return "double_4";
			default: assert(false); return "";
			}
			break;
		default:
			assert(false); return "";
		}
	}

        inline std::string hcHeader()
	{
                return "#include \"hcfftlib.h\"\n"
                       "#include <amp.h>\n"
                       "#include <amp_math.h>\n"
	               "#include <stdio.h>\n"
                       "#include <iostream>\n"
                       "#include <amp_short_vectors.h>\n"
                       "using namespace Concurrency;\n"
                       "using namespace Concurrency::graphics;\n";
	}

	template <Precision PR>
	inline std::string FloatSuffix()
	{
		// Suffix for constants
		std::string sfx;
		switch(PR)
		{
		case P_SINGLE: sfx = "f"; break;
		case P_DOUBLE: sfx = "";  break;
		default: assert(false);
		}

		return sfx;
	}

	inline std::string ButterflyName(size_t radix, size_t count, bool fwd, const hcfftPlanHandle plHandle)
	{
		std::string str;
		if(fwd) str += "Fwd";
		else	str += "Inv";
		str += "Rad"; str += SztToStr(radix);
		str += "B"; str += SztToStr(count);
		str += "H"; str += SztToStr(plHandle);
		return str;
	}

	inline std::string PassName(const hcfftPlanHandle plHandle, size_t pos, bool fwd)
	{
		std::string str;
		if(fwd) str += "Fwd";
		else	str += "Inv";
		str += "Pass";
	        str += SztToStr(plHandle);
		str += SztToStr(pos);
		return str;
	}

	inline std::string TwTableName()
	{
		return "twiddles";
	}

	inline std::string TwTableLargeName()
	{
		return "twiddle_dee";
	}

	inline std::string TwTableLargeFunc()
	{
		return "TW3step";
	}

	// FFT butterfly
       template <Precision PR>
       class Butterfly
       {
	        size_t radix;		// Base radix
                size_t count;       // Number of basic butterflies, valid values: 1,2,4
		bool fwd;			// FFT direction
		bool cReg;			// registers are complex numbers, .x (real), .y(imag)

		size_t BitReverse (size_t n, size_t N) const
		{
			return (N < 2) ? n : (BitReverse (n >> 1, N >> 1) | ((n & 1) != 0 ? (N >> 1) : 0));
		}
		void GenerateButterflyStr(std::string &bflyStr, const hcfftPlanHandle plHandle) const
		{
			std::string regType = cReg ? RegBaseType<PR>(2) : RegBaseType<PR>(count);

			// Function attribute
			bflyStr += "inline void \n";

			// Function name
			bflyStr += ButterflyName(radix, count, fwd, plHandle);

			// Function Arguments
			bflyStr += "(";
			for(size_t i=0;;i++)
			{
				if(cReg)
				{
					bflyStr += regType; bflyStr += " *R";
					if(radix & (radix-1))	bflyStr += SztToStr(i);
					else					bflyStr += SztToStr(BitReverse(i,radix));
				}
				else
				{
					bflyStr += regType; bflyStr += " *R"; bflyStr += SztToStr(i); bflyStr += ", ";	// real arguments
					bflyStr += regType; bflyStr += " *I"; bflyStr += SztToStr(i);					// imaginary arguments
				}

				if(i == radix-1)
				{
					bflyStr += ")";
					break;
				}
				else
				{
					bflyStr += ", ";
				}
			}

			bflyStr += " restrict(amp)\n{\n\n";


			// Temporary variables
			// Allocate temporary variables if we are not using complex registers (cReg = 0) or if cReg is true, then
			// allocate temporary variables only for non power-of-2 radices
			if( (radix & (radix-1)) || (!cReg) )
			{
				bflyStr += "\t";
				if(cReg)
					bflyStr += RegBaseType<PR>(1);
				else
					bflyStr += regType;

				for(size_t i=0;;i++)
				{
					bflyStr += " TR"; bflyStr += SztToStr(i); bflyStr += ",";	// real arguments
					bflyStr += " TI"; bflyStr += SztToStr(i);			// imaginary arguments

					if(i == radix-1)
					{
						bflyStr += ";";
						break;
					}
					else
					{
						bflyStr += ",";
					}
				}
			}
			else
			{
				bflyStr += "\t";
				bflyStr += RegBaseType<PR>(2);
				bflyStr += " T;";
			}


			bflyStr += "\n\n\t";

			// Butterfly for different radices
			switch(radix)
			{
			case 2:
				{
					if(cReg)
					{
						bflyStr +=
						"(R1[0]) = (R0[0]) - (R1[0]);\n\t"
						"(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t";
					}
					else
					{
						bflyStr +=
						"TR0 = (R0[0]) + (R1[0]);\n\t"
						"TI0 = (I0[0]) + (I1[0]);\n\t"
						"TR1 = (R0[0]) - (R1[0]);\n\t"
						"TI1 = (I0[0]) - (I1[0]);\n\t";
					}

				} break;
			case 3:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (R0[0]).x + (R1[0]).x + (R2[0]).x;\n\t"
							"TR1 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) + C3QB*((R1[0]).y - (R2[0]).y);\n\t"
							"TR2 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) - C3QB*((R1[0]).y - (R2[0]).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (R0[0]).y + (R1[0]).y + (R2[0]).y;\n\t"
							"TI1 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) - C3QB*((R1[0]).x - (R2[0]).x);\n\t"
							"TI2 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) + C3QB*((R1[0]).x - (R2[0]).x);\n\t";
						}
						else
						{
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
					}
					else
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (R0[0]).x + (R1[0]).x + (R2[0]).x;\n\t"
							"TR1 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) - C3QB*((R1[0]).y - (R2[0]).y);\n\t"
							"TR2 = ((R0[0]).x - C3QA*((R1[0]).x + (R2[0]).x)) + C3QB*((R1[0]).y - (R2[0]).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (R0[0]).y + (R1[0]).y + (R2[0]).y;\n\t"
							"TI1 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) + C3QB*((R1[0]).x - (R2[0]).x);\n\t"
							"TI2 = ((R0[0]).y - C3QA*((R1[0]).y + (R2[0]).y)) - C3QB*((R1[0]).x - (R2[0]).x);\n\t";
						}
						else
						{
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
				} break;
			case 4:
				{
					if(fwd)
					{
						if(cReg)
						{
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
						}
						else
						{
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
					}
					else
					{
						if(cReg)
						{
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
						}
						else
						{
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
				} break;
			case 5:
				{
					if(fwd)
					{
						if(cReg)
						{
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
						}
						else
						{
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
					}
					else
					{
						if(cReg)
						{
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
						}
						else
						{
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
				} break;
			case 6:
				{
					if(fwd)
					{
						if(cReg)
						{
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
						}
						else
						{
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
					}
					else
					{
						if(cReg)
						{
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
						}
						else
						{
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
				} break;
			case 8:
				{
					if(fwd)
					{
						if(cReg)
						{
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
						}
						else
						{
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
					}
					else
					{
						if(cReg)
						{
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
						}
						else
						{
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
				} break;
			case 10:
				{
					if(fwd)
					{
						if(cReg)
						{
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
						}
						else
						{
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
					}
					else
					{
						if(cReg)
						{
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
						}
						else
						{
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
				} break;
			case 16:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=

							"(R1[0]) = (R0[0]) - (R1[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t"
							"(R3[0]) = (R2[0]) - (R3[0]);\n\t"
							"(R2[0]) = 2.0f * (R2[0]) - (R3[0]);\n\t"
							"(R5[0]) = (R4[0]) - (R5[0]);\n\t"
							"(R4[0]) = 2.0f * (R4[0]) - (R5[0]);\n\t"
							"(R7[0]) = (R6[0]) - (R7[0]);\n\t"
							"(R6[0]) = 2.0f * (R6[0]) - (R7[0]);\n\t"
							"(R9[0]) = (R8[0]) - (R9[0]);\n\t"
							"(R8[0]) = 2.0f * (R8[0]) - (R9[0]);\n\t"
							"(R11[0]) = (R10[0]) - (R11[0]);\n\t"
							"(R10[0]) = 2.0f * (R10[0]) - (R11[0]);\n\t"
							"(R13[0]) = (R12[0]) - (R13[0]);\n\t"
							"(R12[0]) = 2.0f * (R12[0]) - (R13[0]);\n\t"
							"(R15[0]) = (R14[0]) - (R15[0]);\n\t"
							"(R14[0]) = 2.0f * (R14[0]) - (R15[0]);\n\t"
							"\n\t"
							"(R2[0]) = (R0[0]) - (R2[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R2[0]);\n\t"
							"(R3[0]) = (R1[0]) + fvect2(-(R3[0]).y, (R3[0]).x);\n\t"
							"(R1[0]) = 2.0f * (R1[0]) - (R3[0]);\n\t"
							"(R6[0]) = (R4[0]) - (R6[0]);\n\t"
							"(R4[0]) = 2.0f * (R4[0]) - (R6[0]);\n\t"
							"(R7[0]) = (R5[0]) + fvect2(-(R7[0]).y, (R7[0]).x);\n\t"
							"(R5[0]) = 2.0f * (R5[0]) - (R7[0]);\n\t"
							"(R10[0]) = (R8[0]) - (R10[0]);\n\t"
							"(R8[0]) = 2.0f * (R8[0]) - (R10[0]);\n\t"
							"(R11[0]) = (R9[0]) + fvect2(-(R11[0]).y, (R11[0]).x);\n\t"
							"(R9[0]) = 2.0f * (R9[0]) - (R11[0]);\n\t"
							"(R14[0]) = (R12[0]) - (R14[0]);\n\t"
							"(R12[0]) = 2.0f * (R12[0]) - (R14[0]);\n\t"
							"(R15[0]) = (R13[0]) + fvect2(-(R15[0]).y, (R15[0]).x);\n\t"
							"(R13[0]) = 2.0f * (R13[0]) - (R15[0]);\n\t"
							"\n\t"
							"(R4[0]) = (R0[0]) - (R4[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R4[0]);\n\t"
							"(R5[0]) = ((R1[0]) - C8Q * (R5[0])) - C8Q * fvect2((R5[0]).y, -(R5[0]).x);\n\t"
							"(R1[0]) = 2.0f * (R1[0]) - (R5[0]);\n\t"
							"(R6[0]) = (R2[0]) + fvect2(-(R6[0]).y, (R6[0]).x);\n\t"
							"(R2[0]) = 2.0f * (R2[0]) - (R6[0]);\n\t"
							"(R7[0]) = ((R3[0]) + C8Q * (R7[0])) - C8Q * fvect2((R7[0]).y, -(R7[0]).x);\n\t"
							"(R3[0]) = 2.0f * (R3[0]) - (R7[0]);\n\t"
							"(R12[0]) = (R8[0]) - (R12[0]);\n\t"
							"(R8[0]) = 2.0f * (R8[0]) - (R12[0]);\n\t"
							"(R13[0]) = ((R9[0]) - C8Q * (R13[0])) - C8Q * fvect2((R13[0]).y, -(R13[0]).x);\n\t"
							"(R9[0]) = 2.0f * (R9[0]) - (R13[0]);\n\t"
							"(R14[0]) = (R10[0]) + fvect2(-(R14[0]).y, (R14[0]).x);\n\t"
							"(R10[0]) = 2.0f * (R10[0]) - (R14[0]);\n\t"
							"(R15[0]) = ((R11[0]) + C8Q * (R15[0])) - C8Q * fvect2((R15[0]).y, -(R15[0]).x);\n\t"
							"(R11[0]) = 2.0f * (R11[0]) - (R15[0]);\n\t"
							"\n\t"
							"(R8[0]) = (R0[0]) - (R8[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R8[0]);\n\t"
							"(R9[0]) = ((R1[0]) - 0.92387953251128675612818318939679 * (R9[0])) - 0.3826834323650897717284599840304 * fvect2((R9[0]).y, -(R9[0]).x);\n\t"
							"(R1[0]) = 2.0f * (R1[0]) - (R9[0]);\n\t"
							"(R10[0]) = ((R2[0]) - C8Q * (R10[0])) - C8Q * fvect2((R10[0]).y, -(R10[0]).x);\n\t"
							"(R2[0]) = 2.0f * (R2[0]) - (R10[0]);\n\t"
							"(R11[0]) = ((R3[0]) - 0.3826834323650897717284599840304 * (R11[0])) - 0.92387953251128675612818318939679 * fvect2((R11[0]).y, -(R11[0]).x);\n\t"
							"(R3[0]) = 2.0f * (R3[0]) - (R11[0]);\n\t"
							"(R12[0]) = (R4[0]) + fvect2(-(R12[0]).y, (R12[0]).x);\n\t"
							"(R4[0]) = 2.0f * (R4[0]) - (R12[0]);\n\t"
							"(R13[0]) = ((R5[0]) + 0.3826834323650897717284599840304 * (R13[0])) - 0.92387953251128675612818318939679 * fvect2((R13[0]).y, -(R13[0]).x);\n\t"
							"(R5[0]) = 2.0f * (R5[0]) - (R13[0]);\n\t"
							"(R14[0]) = ((R6[0]) + C8Q * (R14[0])) - C8Q * fvect2((R14[0]).y, -(R14[0]).x);\n\t"
							"(R6[0]) = 2.0f * (R6[0]) - (R14[0]);\n\t"
							"(R15[0]) = ((R7[0]) + 0.92387953251128675612818318939679 * (R15[0])) - 0.3826834323650897717284599840304 * fvect2((R15[0]).y, -(R15[0]).x);\n\t"
							"(R7[0]) = 2.0f * (R7[0]) - (R15[0]);\n\t";

						}
						else
							assert(false);
					}
					else
					{
						if(cReg)
						{
							bflyStr +=

							"(R1[0]) = (R0[0]) - (R1[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R1[0]);\n\t"
							"(R3[0]) = (R2[0]) - (R3[0]);\n\t"
							"(R2[0]) = 2.0f * (R2[0]) - (R3[0]);\n\t"
							"(R5[0]) = (R4[0]) - (R5[0]);\n\t"
							"(R4[0]) = 2.0f * (R4[0]) - (R5[0]);\n\t"
							"(R7[0]) = (R6[0]) - (R7[0]);\n\t"
							"(R6[0]) = 2.0f * (R6[0]) - (R7[0]);\n\t"
							"(R9[0]) = (R8[0]) - (R9[0]);\n\t"
							"(R8[0]) = 2.0f * (R8[0]) - (R9[0]);\n\t"
							"(R11[0]) = (R10[0]) - (R11[0]);\n\t"
							"(R10[0]) = 2.0f * (R10[0]) - (R11[0]);\n\t"
							"(R13[0]) = (R12[0]) - (R13[0]);\n\t"
							"(R12[0]) = 2.0f * (R12[0]) - (R13[0]);\n\t"
							"(R15[0]) = (R14[0]) - (R15[0]);\n\t"
							"(R14[0]) = 2.0f * (R14[0]) - (R15[0]);\n\t"
							"\n\t"
							"(R2[0]) = (R0[0]) - (R2[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R2[0]);\n\t"
							"(R3[0]) = (R1[0]) + fvect2((R3[0]).y, -(R3[0]).x);\n\t"
							"(R1[0]) = 2.0f * (R1[0]) - (R3[0]);\n\t"
							"(R6[0]) = (R4[0]) - (R6[0]);\n\t"
							"(R4[0]) = 2.0f * (R4[0]) - (R6[0]);\n\t"
							"(R7[0]) = (R5[0]) + fvect2((R7[0]).y, -(R7[0]).x);\n\t"
							"(R5[0]) = 2.0f * (R5[0]) - (R7[0]);\n\t"
							"(R10[0]) = (R8[0]) - (R10[0]);\n\t"
							"(R8[0]) = 2.0f * (R8[0]) - (R10[0]);\n\t"
							"(R11[0]) = (R9[0]) + fvect2((R11[0]).y, -(R11[0]).x);\n\t"
							"(R9[0]) = 2.0f * (R9[0]) - (R11[0]);\n\t"
							"(R14[0]) = (R12[0]) - (R14[0]);\n\t"
							"(R12[0]) = 2.0f * (R12[0]) - (R14[0]);\n\t"
							"(R15[0]) = (R13[0]) + fvect2((R15[0]).y, -(R15[0]).x);\n\t"
							"(R13[0]) = 2.0f * (R13[0]) - (R15[0]);\n\t"
							"\n\t"
							"(R4[0]) = (R0[0]) - (R4[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R4[0]);\n\t"
							"(R5[0]) = ((R1[0]) - C8Q * (R5[0])) + C8Q * fvect2((R5[0]).y, -(R5[0]).x);\n\t"
							"(R1[0]) = 2.0f * (R1[0]) - (R5[0]);\n\t"
							"(R6[0]) = (R2[0]) + fvect2((R6[0]).y, -(R6[0]).x);\n\t"
							"(R2[0]) = 2.0f * (R2[0]) - (R6[0]);\n\t"
							"(R7[0]) = ((R3[0]) + C8Q * (R7[0])) + C8Q * fvect2((R7[0]).y, -(R7[0]).x);\n\t"
							"(R3[0]) = 2.0f * (R3[0]) - (R7[0]);\n\t"
							"(R12[0]) = (R8[0]) - (R12[0]);\n\t"
							"(R8[0]) = 2.0f * (R8[0]) - (R12[0]);\n\t"
							"(R13[0]) = ((R9[0]) - C8Q * (R13[0])) + C8Q * fvect2((R13[0]).y, -(R13[0]).x);\n\t"
							"(R9[0]) = 2.0f * (R9[0]) - (R13[0]);\n\t"
							"(R14[0]) = (R10[0]) + fvect2((R14[0]).y, -(R14[0]).x);\n\t"
							"(R10[0]) = 2.0f * (R10[0]) - (R14[0]);\n\t"
							"(R15[0]) = ((R11[0]) + C8Q * (R15[0])) + C8Q * fvect2((R15[0]).y, -(R15[0]).x);\n\t"
							"(R11[0]) = 2.0f * (R11[0]) - (R15[0]);\n\t"
 							"\n\t"
							"(R8[0]) = (R0[0]) - (R8[0]);\n\t"
							"(R0[0]) = 2.0f * (R0[0]) - (R8[0]);\n\t"
							"(R9[0]) = ((R1[0]) - 0.92387953251128675612818318939679 * (R9[0])) + 0.3826834323650897717284599840304 * fvect2((R9[0]).y, -(R9[0]).x);\n\t"
							"(R1[0]) = 2.0f * (R1[0]) - (R9[0]);\n\t"
							"(R10[0]) = ((R2[0]) - C8Q * (R10[0])) + C8Q * fvect2((R10[0]).y, -(R10[0]).x);\n\t"
							"(R2[0]) = 2.0f * (R2[0]) - (R10[0]);\n\t"
							"(R11[0]) = ((R3[0]) - 0.3826834323650897717284599840304 * (R11[0])) + 0.92387953251128675612818318939679 * fvect2((R11[0]).y, -(R11[0]).x);\n\t"
							"(R3[0]) = 2.0f * (R3[0]) - (R11[0]);\n\t"
							"(R12[0]) = (R4[0]) + fvect2((R12[0]).y, -(R12[0]).x);\n\t"
							"(R4[0]) = 2.0f * (R4[0]) - (R12[0]);\n\t"
							"(R13[0]) = ((R5[0]) + 0.3826834323650897717284599840304 * (R13[0])) + 0.92387953251128675612818318939679 * fvect2((R13[0]).y, -(R13[0]).x);\n\t"
							"(R5[0]) = 2.0f * (R5[0]) - (R13[0]);\n\t"
							"(R14[0]) = ((R6[0]) + C8Q * (R14[0])) + C8Q * fvect2((R14[0]).y, -(R14[0]).x);\n\t"
							"(R6[0]) = 2.0f * (R6[0]) - (R14[0]);\n\t"
							"(R15[0]) = ((R7[0]) + 0.92387953251128675612818318939679 * (R15[0])) + 0.3826834323650897717284599840304 * fvect2((R15[0]).y, -(R15[0]).x);\n\t"
							"(R7[0]) = 2.0f * (R7[0]) - (R15[0]);\n\t";

						}
						else
							assert(false);
					}
				} break;
			default:
				assert(false);
			}

			bflyStr += "\n\t";

			// Assign results
			if( (radix & (radix-1)) || (!cReg) )
			{
				if( (radix != 10) && (radix != 6) )
				{
				for(size_t i=0; i<radix;i++)
				{
					if(cReg)
					{
						bflyStr += "((R"; bflyStr += SztToStr(i); bflyStr += "[0]).x) = TR"; bflyStr += SztToStr(i); bflyStr += "; ";
						bflyStr += "((R"; bflyStr += SztToStr(i); bflyStr += "[0]).y) = TI"; bflyStr += SztToStr(i); bflyStr += ";\n\t";
					}
					else
					{
						bflyStr += "(R"; bflyStr += SztToStr(i); bflyStr += "[0]) = TR"; bflyStr += SztToStr(i); bflyStr += "; ";
						bflyStr += "(I"; bflyStr += SztToStr(i); bflyStr += "[0]) = TI"; bflyStr += SztToStr(i); bflyStr += ";\n\t";
					}
				}
				}
			}
			else
			{
				for(size_t i=0; i<radix;i++)
				{
					size_t j = BitReverse(i, radix);

					if(i < j)
					{
						bflyStr += "T = (R"; bflyStr += SztToStr(i); bflyStr += "[0]); (R";
						bflyStr += SztToStr(i); bflyStr += "[0]) = (R"; bflyStr += SztToStr(j); bflyStr += "[0]); (R";
						bflyStr += SztToStr(j); bflyStr += "[0]) = T;\n\t";
					}
				}
			}

			bflyStr += "\n}\n";
		}
	public:
		Butterfly(size_t radixVal, size_t countVal, bool fwdVal, bool cRegVal) : radix(radixVal), count(countVal), fwd(fwdVal), cReg(cRegVal) {}

		void GenerateButterfly(std::string &bflyStr, const hcfftPlanHandle plHandle) const
		{
			assert(count <= 4);
			if(count > 0)
				GenerateButterflyStr(bflyStr, plHandle);
		}
       };
}
