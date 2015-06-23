#include <math.h>
#include "stockham.h"
#include <list>
// FFT Stockham Autosort Method
//
//   Each pass does one digit reverse in essence. Hence by the time all passes are done, complete
//   digit reversal is done and output FFT is in correct order. Intermediate FFTs are stored in natural order,
//   which is not the case with basic Cooley-Tukey algorithm. Natural order in intermediate data makes it
//   convenient for stitching together passes with different radices.
//
//  Basic FFT algorithm:
//
//        Pass loop
//        {
//            Outer loop
//            {
//                Inner loop
//                {
//                }
//            }
//        }
//
//  The sweeps of the outer and inner loop resemble matrix indexing, this matrix changes shape with every pass as noted below
//
//   FFT pass diagram (radix 2)
//
//                k            k+R                                    k
//            * * * * * * * * * * * * * * * *                     * * * * * * * *
//            *   |             |           *                     *   |         *
//            *   |             |           *                     *   |         *
//            *   |             |           * LS        -->       *   |         *
//            *   |             |           *                     *   |         *
//            *   |             |           *                     *   |         *
//            * * * * * * * * * * * * * * * *                     *   |         *
//                         RS                                     *   |         * L
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                *   |         *
//                                                                * * * * * * * *
//                                                                       R
//
//
//    With every pass, the matrix doubles in height and halves in length
//
//
//  N = 2^T = Length of FFT
//  q = pass loop index
//  k = outer loop index = (0 ... R-1)
//  j = inner loop index = (0 ... LS-1)
//
//  Tables shows how values change as we go through the passes
//
//    q | LS   |  R   |  L  | RS
//   ___|______|______|_____|___
//    0 |  1   | N/2  |  2  | N
//    1 |  2   | N/4  |  4  | N/2
//    2 |  4   | N/8  |  8  | N/4
//    . |  .   | .    |  .  | .
//  T-1 |  N/2 | 1    |  N  | 2
//
//
//   Data Read Order
//     Radix 2: k*LS + j, (k+R)*LS + j
//     Radix 3: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j
//     Radix 4: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j
//     Radix 5: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j, (k+4R)*LS + j
//
//   Data Write Order
//       Radix 2: k*L + j, k*L + j + LS
//       Radix 3: k*L + j, k*L + j + LS, k*L + j + 2*LS
//       Radix 4: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS
//       Radix 5: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS, k*L + j + 4*LS
//

namespace StockhamGenerator
{
// Experimental End ===========================================

#define RADIX_TABLE_COMMON 	{     2048,           256,             1,         4,     8, 8, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{      512,            64,             1,         3,     8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{      256,            64,             1,         4,     4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       64,            64,             4,         3,     4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       32,            64,            16,         2,     8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       16,            64,            16,         2,     4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{        4,            64,            32,         2,     2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{        2,            64,            64,         1,     2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

    template <Precision PR>
	class KernelCoreSpecs
	{
		struct SpecRecord
		{
			size_t length;
			size_t workGroupSize;
			size_t numTransforms;
			size_t numPasses;
			size_t radices[12]; // Setting upper limit of number of passes to 12
		};

		typedef typename std::map<size_t, SpecRecord> SpecTable;
		SpecTable specTable;

	public:
		KernelCoreSpecs()
		{
			switch(PR)
			{
			case P_SINGLE:
				{
					SpecRecord specRecord[] = {

					RADIX_TABLE_COMMON

					//  Length, WorkGroupSize, NumTransforms, NumPasses,  Radices
					{     4096,           256,             1,         4,     8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0 },
					{     1024,           128,             1,         4,     8, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
					{      128,            64,             4,         3,     8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
					{        8,            64,            32,         2,     4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

					};

					size_t tableLength = sizeof(specRecord)/sizeof(specRecord[0]);
					for(size_t i=0; i<tableLength; i++) specTable[specRecord[i].length] = specRecord[i];

				} break;

			case P_DOUBLE:
				{
					SpecRecord specRecord[] = {

					RADIX_TABLE_COMMON

					//  Length, WorkGroupSize, NumTransforms, NumPasses,  Radices
					{     1024,           128,             1,         4,     8, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
					//{      128,            64,             1,         7,     2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0 },
					{      128,            64,             4,         3,     8, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
					{        8,            64,            16,         3,     2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

					};

					size_t tableLength = sizeof(specRecord)/sizeof(specRecord[0]);
					for(size_t i=0; i<tableLength; i++) specTable[specRecord[i].length] = specRecord[i];
				} break;

			default:
				assert(false);
			}
		}

		void GetRadices(size_t length, size_t &numPasses, const size_t * &pRadices) const
		{
			pRadices = NULL;
			numPasses = 0;

			typename SpecTable::const_iterator it = specTable.find(length);
			if(it != specTable.end())
			{
				pRadices = it->second.radices;
				numPasses = it->second.numPasses;
			}
		}

		void GetWGSAndNT(size_t length, size_t &workGroupSize, size_t &numTransforms) const
		{
			workGroupSize = 0;
			numTransforms = 0;

			typename SpecTable::const_iterator it = specTable.find(length);
			if(it != specTable.end())
			{
				workGroupSize = it->second.workGroupSize;
				numTransforms = it->second.numTransforms;
			}
		}
	};

	// Given the length of 1d fft, this function determines the appropriate work group size
	// and the number of transforms per work group
	// TODO for optimizations - experiment with different possibilities for work group sizes and num transforms for improving performance
	void DetermineSizes(const size_t &MAX_WGS, const size_t &length, size_t &workGroupSize, size_t &numTrans)
	{
		assert(MAX_WGS >= 64);

		if(length == 1) // special case
		{
			workGroupSize = 64;
			numTrans = 64;
			return;
		}

		size_t baseRadix[] = {5,3,2}; // list only supported primes
		size_t baseRadixSize = sizeof(baseRadix)/sizeof(baseRadix[0]);

		size_t l = length;
		std::map<size_t, size_t> primeFactors;
		std::map<size_t, size_t> primeFactorsExpanded;
		for(size_t r=0; r<baseRadixSize; r++)
		{
			size_t rad = baseRadix[r];
			size_t p = 0;
			size_t e = 1;
			while(!(l%rad))
			{
				l /= rad;
				e *= rad;
				p++;
			}

			primeFactors[rad] = p;
			primeFactorsExpanded[rad] = e;
		}

		assert(l == 1); // Makes sure the number is composed of only supported primes

		if (primeFactorsExpanded[2] == length)	// Length is pure power of 2
		{
			//if(length == 1024) { workGroupSize = 128;  numTrans = 1; }
			if		(length >= 1024)	{ workGroupSize = (MAX_WGS >= 256) ? 256 : MAX_WGS; numTrans = 1; }
			//else if (length == 512)		{ workGroupSize = (MAX_WGS >= 128) ? 128 : MAX_WGS; numTrans = 1; }
			else if (length == 512)		{ workGroupSize = 64; numTrans = 1; }
			else if	(length >= 16)		{ workGroupSize = 64;  numTrans = 256/length; }
			else						{ workGroupSize = 64;  numTrans = 128/length; }
		}
		else if	(primeFactorsExpanded[3] == length) // Length is pure power of 3
		{
			workGroupSize = (MAX_WGS >= 256) ? 243 : 27;
			if(length >= 3*workGroupSize)	numTrans = 1;
			else							numTrans = (3*workGroupSize)/length;
		}
		else if	(primeFactorsExpanded[5] == length) // Length is pure power of 5
		{
			workGroupSize = (MAX_WGS >= 128) ? 125 : 25;
			if(length >= 5*workGroupSize)	numTrans = 1;
			else							numTrans = (5*workGroupSize)/length;
		}
		else
		{
			size_t leastNumPerWI; // least number of elements in one work item
			size_t maxWorkGroupSize; // maximum work group size desired

			if		(primeFactorsExpanded[2] * primeFactorsExpanded[3] == length) // Length is mix of 2&3 only
			{
				if(!(length%12))	{ leastNumPerWI = 12; maxWorkGroupSize = (MAX_WGS >= 128) ? 128 : MAX_WGS; }
				else				{ leastNumPerWI = 6;  maxWorkGroupSize = (MAX_WGS >= 256) ? 256 : MAX_WGS; }
			}
			else if	(primeFactorsExpanded[2] * primeFactorsExpanded[5] == length) // Length is mix of 2&5 only
			{
				if(!(length%20))	{ leastNumPerWI = 20; maxWorkGroupSize = 64; }
				else				{ leastNumPerWI = 10; maxWorkGroupSize = (MAX_WGS >= 128) ? 128 : MAX_WGS; }
			}
			else if (primeFactorsExpanded[3] * primeFactorsExpanded[5] == length) // Length is mix of 3&5 only
			{
				leastNumPerWI = 15;
				maxWorkGroupSize = 64;
			}
			else
			{
				leastNumPerWI = 30;
				maxWorkGroupSize = 64;
			}


			// Make sure the work group size does not exceed MAX_WGS
			// for large problems sizes, this means doing more work per work-item
			size_t lnpi;
			size_t ft = 1;
			while(1)
			{
				lnpi = leastNumPerWI * ft++;
				if(length%lnpi) continue;

				if( (length/lnpi) <= MAX_WGS )
				{
					leastNumPerWI = lnpi;
					break;
				}
			}

			numTrans = 1;
			size_t n=1;
			while( ((n*length)/leastNumPerWI) <= maxWorkGroupSize )
			{
				numTrans = n;
				n++;
			}

			workGroupSize = (numTrans*length)/leastNumPerWI;
			assert(workGroupSize <= MAX_WGS);
		}
	}
	// Twiddle factors table
       class TwiddleTable
       {
                size_t N; // length
	        double *wc, *ws; // cosine, sine arrays

	public:
		TwiddleTable(size_t length) : N(length)
		{
			// Allocate memory for the tables
			// We compute twiddle factors in double precision for both P_SINGLE and P_DOUBLE
			wc = new double[N];
			ws = new double[N];
		}

		~TwiddleTable()
		{
			// Free
			delete[] wc;
			delete[] ws;
		}

		template <Precision PR>
		void GenerateTwiddleTable(const std::vector<size_t> &radices, std::string &twStr)
		{
			const double TWO_PI = -6.283185307179586476925286766559;

			// Make sure the radices vector sums up to N
			size_t sz = 1;
			for(std::vector<size_t>::const_iterator i = radices.begin();
				i != radices.end(); i++)
			{
				sz *= (*i);
			}
			assert(sz == N);
			// Generate the table
			size_t L = 1;
			size_t nt = 0;
			for(std::vector<size_t>::const_iterator i = radices.begin();
				i != radices.end(); i++)
			{
				size_t radix = *i;

				L *= radix;

				// Twiddle factors
				for(size_t k=0; k<(L/radix); k++)
				{
					double theta = TWO_PI * ((double)k)/((double)L);

					for(size_t j=1; j<radix; j++)
					{
						double c = cos(((double)j) * theta);
						double s = sin(((double)j) * theta);

						//if (fabs(c) < 1.0E-12)	c = 0.0;
						//if (fabs(s) < 1.0E-12)	s = 0.0;

						wc[nt]   = c;
						ws[nt++] = s;
					}
				}
			}

			std::string sfx = FloatSuffix<PR>();

			// Stringize the table
			std::stringstream ss;
			for(size_t i = 0; i < (N-1); i++)
			{
				ss << "("; ss << RegBaseType<PR>(2); ss << ")(";

				char cv[64], sv[64];
				sprintf(cv, "%036.34lf", wc[i]);
				sprintf(sv, "%036.34lf", ws[i]);
				ss << cv; ss << sfx; ss << ", ";
				ss << sv; ss << sfx; ss << "),\n";
			}
			twStr += ss.str();
		}
    };

	// Twiddle factors table for large N
	// used in 3-step algorithm
    class TwiddleTableLarge
    {
        size_t N; // length
		size_t X, Y;
		size_t tableSize;
		double *wc, *ws; // cosine, sine arrays

	public:
		TwiddleTableLarge(size_t length) : N(length)
		{
			X = size_t(1) << ARBITRARY::TWIDDLE_DEE;
			Y = DivRoundingUp<size_t> (CeilPo2(N), ARBITRARY::TWIDDLE_DEE);
			tableSize = X * Y;

			// Allocate memory for the tables
			wc = new double[tableSize];
			ws = new double[tableSize];
		}

		~TwiddleTableLarge()
		{
			// Free
			delete[] wc;
			delete[] ws;
		}

		template <Precision PR>
		void GenerateTwiddleTable(std::string &twStr)
		{
			const double TWO_PI = -6.283185307179586476925286766559;

			// Generate the table
			size_t nt = 0;
			double phi = TWO_PI / double (N);
			for (size_t iY = 0; iY < Y; ++iY)
			{
				size_t i = size_t(1) << (iY * ARBITRARY::TWIDDLE_DEE);
				for (size_t iX = 0; iX < X; ++iX)
				{
					size_t j = i * iX;

					double c = cos(phi * (double)j);
					double s = sin(phi * (double)j);

					//if (fabs(c) < 1.0E-12)	c = 0.0;
					//if (fabs(s) < 1.0E-12)	s = 0.0;

					wc[nt]   = c;
					ws[nt++] = s;
				}
			}

			std::string sfx = FloatSuffix<PR>();

			// Stringize the table
			std::stringstream ss;
			nt = 0;

			ss << "\n const ";
			ss << RegBaseType<PR>(2);
			ss << " " << TwTableLargeName();
			ss << "[" << Y << "][" << X << "] = {\n";
			for (size_t iY = 0; iY < Y; ++iY)
			{
				ss << "{ ";
				for (size_t iX = 0; iX < X; ++iX)
				{
					char cv[64], sv[64];
					sprintf(cv, "%036.34lf", wc[nt]);
					sprintf(sv, "%036.34lf", ws[nt++]);
					ss << "("; ss << RegBaseType<PR>(2); ss << ")(";
					ss << cv; ss << sfx; ss << ", ";
					ss << sv; ss << sfx; ss << ")";
					ss << ", ";
				}
				ss << " },\n";
			}
			ss << "};\n\n";


			// Twiddle calc function
			ss << "inline void ";
			ss << RegBaseType<PR>(2);
			ss << "\n" << TwTableLargeFunc() << "(unsigned int u)\n{\n";

			ss << "\t" "unsigned int j = u & " << unsigned(X-1) << ";\n";
			ss << "\t" ; ss << RegBaseType<PR>(2); ss << " result = ";
			ss << TwTableLargeName();
			ss << "[0][j];\n";

			for (size_t iY = 1; iY < Y; ++iY)
			{
				std::string phasor = TwTableLargeName();
				phasor += "[";
				phasor += SztToStr(iY);
				phasor += "][j]";

				stringpair product = ComplexMul((RegBaseType<PR>(2)).c_str(), "result", phasor.c_str());

				ss << "\t" "u >>= " << unsigned (ARBITRARY::TWIDDLE_DEE) << ";\n";
				ss << "\t" "j = u & " << unsigned(X-1) << ";\n";
				ss << "\t" "result = " << product.first << "\n";
				ss << "\t" "\t" << product.second <<";\n";
			}
			ss << "\t" "return result;\n}\n\n";

			twStr += ss.str();
		}
    };

    // A pass inside an FFT kernel
    template <Precision PR>
    class Pass
    {
		size_t position;					// Position in the kernel

		size_t algL;						// 'L' value from fft algorithm
		size_t algLS;						// 'LS' value
		size_t algR;						// 'R' value

		size_t length;						// Length of FFT
        size_t radix;						// Base radix
		size_t cnPerWI;						// Complex numbers per work-item

		size_t workGroupSize;				// size of the workgroup = (length / cnPerWI)
											// this number is essentially number of work-items needed to compute 1 transform
											// this number will be different from the kernel class workGroupSize if there
											// are multiple transforms per workgroup

		size_t numButterfly;				// Number of basic FFT butterflies = (cnPerWI / radix)
		size_t numB1, numB2, numB4;			// number of different types of butterflies

		bool r2c;							// real to complex transform
		bool c2r;							// complex to real transform
		bool rcFull;
		bool rcSimple;

		bool enableGrouping;
		bool linearRegs;
		Pass<PR> *nextPass;

		inline void RegBase(size_t regC, std::string &str) const
		{
			str += "B";
			str += SztToStr(regC);
		}

		inline void RegBaseAndCount(size_t num, std::string &str) const
		{
			str += "C";
			str += SztToStr(num);
		}

		inline void RegBaseAndCountAndPos(const std::string &RealImag, size_t radPos, std::string &str) const
		{
			str += RealImag;
			str += SztToStr(radPos);
		}

		void RegIndex(size_t regC, size_t num, const std::string &RealImag, size_t radPos, std::string &str) const
		{
			RegBase(regC, str);
			RegBaseAndCount(num, str);
			RegBaseAndCountAndPos(RealImag, radPos, str);
		}

		void DeclareRegs(const std::string &regType, size_t regC, size_t numB, std::string &passStr) const
		{
			std::string regBase;
			RegBase(regC, regBase);

			if(linearRegs)
			{
				assert(regC == 1);
				assert(numB == numButterfly);
			}

			for(size_t i=0; i<numB; i++)
			{
				passStr += "\n\t";
				passStr += regType;
				passStr += " ";

				std::string regBaseCount = regBase;
				RegBaseAndCount(i, regBaseCount);

				for(size_t r=0; ; r++)
				{
					if(linearRegs)
					{
						std::string regIndex = "R";
						RegBaseAndCountAndPos("", i*radix + r, regIndex);

						passStr += regIndex;
					}
					else
					{
						std::string regRealIndex(regBaseCount), regImagIndex(regBaseCount);

						RegBaseAndCountAndPos("R", r, regRealIndex); // real
						RegBaseAndCountAndPos("I", r, regImagIndex); // imaginary

						passStr += regRealIndex; passStr += ", ";
						passStr += regImagIndex;
					}

					if(r == radix-1)
					{
						passStr += ";";
						break;
					}
					else
					{
						passStr += ", ";
					}
				}
			}
		}

		inline std::string IterRegArgs() const
		{
			std::string str = "";

			if(linearRegs)
			{
				std::string regType = RegBaseType<PR>(2);

				for(size_t i=0; i<cnPerWI; i++)
				{
					if(i != 0) str += ", ";
					str += regType; str += " *R";
					str += SztToStr(i);
				}
			}

			return str;
		}

#define SR_READ			1
#define SR_TWMUL		2
#define SR_TWMUL_3STEP	3
#define SR_WRITE		4

#define SR_COMP_REAL 0 // real
#define SR_COMP_IMAG 1 // imag
#define SR_COMP_BOTH 2 // real & imag

		// SweepRegs is to iterate through the registers to do the three basic operations:
		// reading, twiddle multiplication, writing
		void SweepRegs(	size_t flag, bool fwd, bool interleaved, size_t stride, size_t component,
						double scale,
						const std::string &bufferRe, const std::string &bufferIm, const std::string &offset,
						size_t regC, size_t numB, size_t numPrev, std::string &passStr) const
		{
			assert( (flag == SR_READ )			||
					(flag == SR_TWMUL)			||
					(flag == SR_TWMUL_3STEP)	||
					(flag == SR_WRITE) );

			const std::string twTable = TwTableName();
			const std::string tw3StepFunc = TwTableLargeFunc();

			// component: 0 - real, 1 - imaginary, 2 - both
			size_t cStart, cEnd;
			switch(component)
			{
			case SR_COMP_REAL:	cStart = 0; cEnd = 1; break;
			case SR_COMP_IMAG:	cStart = 1; cEnd = 2; break;
			case SR_COMP_BOTH:	cStart = 0; cEnd = 2; break;
			default:	assert(false);
			}

			// Read/Write logic:
			// The double loop inside pass loop of FFT algorithm is mapped into the
			// workGroupSize work items with each work item handling cnPerWI numbers

			// Read logic:
			// Reads for any pass appear the same with the stockham algorithm when mapped to
			// the work items. The buffer is divided into (L/radix) sized blocks and the
			// values are read in linear order inside each block.

			// Vector reads are possible if we have unit strides
			// since read pattern remains the same for all passes and they are contiguous
			// Writes are not contiguous

			// TODO : twiddle multiplies can be combined with read
			// TODO : twiddle factors can be reordered in the table to do vector reads of them

			// Write logic:
			// outer loop index k and the inner loop index j map to 'me' as follows:
			// In one work-item (1 'me'), there are 'numButterfly' fft butterflies. They
			// are indexed as numButterfly*me + butterflyIndex, where butterflyIndex's range is
			// 0 ... numButterfly-1. The total number of butterflies needed is covered over all
			// the work-items. So essentially the double loop k,j is flattened to fit this linearly
			// increasing 'me'.
			// j = (numButterfly*me + butterflyIndex)%LS
			// k = (numButterfly*me + butterflyIndex)/LS


			std::string twType = RegBaseType<PR>(2);
			std::string rType  = RegBaseType<PR>(1);

			size_t butterflyIndex = numPrev;

			std::string regBase;
			RegBase(regC, regBase);

			// special write back to global memory with float4 grouping, writing 2 complex numbers at once
			if( numB && (numB%2 == 0) && (regC == 1) && (stride == 1) && (numButterfly%2 == 0) && (algLS%2 == 0) && (flag == SR_WRITE) &&
				(nextPass == NULL) && interleaved && (component == SR_COMP_BOTH) && linearRegs && enableGrouping )
			{
				assert((numButterfly * workGroupSize) == algLS);
				assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag buffer strings are same for interleaved data

				passStr += "\n\t";
				passStr += "const array_view<"; passStr += RegBaseType<PR>(4);
				passStr += ",1> &buff4g = "; passStr += bufferRe; passStr += ";\n\t"; // Assuming 'outOffset' is 0, so not adding it here

				for(size_t r=0; r<radix; r++) // setting the radix loop outside to facilitate grouped writing
				{
					butterflyIndex = numPrev;

					for(size_t i=0; i<(numB/2); i++)
					{
						std::string regIndexA = "(*R";
						std::string regIndexB = "(*R";

						RegBaseAndCountAndPos("", (2*i + 0)*radix + r, regIndexA); regIndexA += ")";
						RegBaseAndCountAndPos("", (2*i + 1)*radix + r, regIndexB); regIndexB += ")";

						passStr += "\n\t";
						passStr += "buff4g"; passStr += "[ ";
						passStr += SztToStr(numButterfly/2); passStr += "*me + "; passStr += SztToStr(butterflyIndex);
						passStr += " + ";
						passStr += SztToStr(r*(algLS/2)); passStr += " ]";
						passStr += " = "; passStr += "("; passStr += RegBaseType<PR>(4); passStr += ")(";
						passStr += regIndexA; passStr += ".x, ";
						passStr += regIndexA; passStr += ".y, ";
						passStr += regIndexB; passStr += ".x, ";
						passStr += regIndexB; passStr += ".y) ";
						if(scale != 1.0f) { passStr += " * "; passStr += FloatToStr(scale); passStr += FloatSuffix<PR>(); }
						passStr += ";";

						butterflyIndex++;
					}
				}

				return;
			}

			for(size_t i=0; i<numB; i++)
			{
				std::string regBaseCount = regBase;
				RegBaseAndCount(i, regBaseCount);

				if(flag == SR_READ) // read operation
				{
					// the 'r' (radix index) loop is placed outer to the
					// 'v' (vector index) loop to make possible vectorized reads

					for(size_t r=0; r<radix; r++)
					{
						for(size_t c=cStart; c<cEnd; c++) // component loop: 0 - real, 1 - imaginary
						{
							std::string tail;
							std::string regIndex;
							regIndex = linearRegs ? "(R" : regBaseCount;
                                                        std::string buffer;

							// Read real & imag at once
							if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
							{
								assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag buffer strings are same for interleaved data
								buffer = bufferRe;
								RegBaseAndCountAndPos("", i*radix + r, regIndex); regIndex += "[0])";
								tail = ";";
							}
							else
							{
								if(c == 0)
								{
									if(linearRegs) { RegBaseAndCountAndPos("", i*radix + r, regIndex); regIndex += "[0]).x"; }
									else		   { RegBaseAndCountAndPos("R", r, regIndex); }
									buffer = bufferRe;
									tail = interleaved ? ".x;" : ";";
								}
								else
								{
									if(linearRegs) { RegBaseAndCountAndPos("", i*radix + r, regIndex); regIndex += "[0]).y"; }
									else		   { RegBaseAndCountAndPos("I", r, regIndex); }
									buffer = bufferIm;
									tail = interleaved ? ".y;" : ";";
								}
							}

							for(size_t v=0; v<regC; v++) // TODO: vectorize the reads; instead of reading individually for consecutive reads of vector elements
							{
								std::string regIndexSub(regIndex);
								if(regC != 1)
								{
									regIndexSub += ".s";
									regIndexSub += SztToStr(v);
								}

								passStr += "\n\t";
								passStr += regIndexSub;
								passStr += " = "; passStr += buffer;
								passStr += "["; passStr += offset; passStr += " + ( "; passStr += SztToStr(numPrev); passStr += " + ";
								passStr += "me*"; passStr += SztToStr(numButterfly); passStr += " + ";
								passStr += SztToStr(i*regC + v); passStr += " + ";
								passStr += SztToStr(r*length/radix); passStr += " )*";
								passStr += SztToStr(stride); passStr += "]"; passStr += tail;
							}

							// Since we read real & imag at once, we break the loop
							if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
								break;
						}
					}
				}
				else if( (flag == SR_TWMUL) || (flag == SR_TWMUL_3STEP) ) // twiddle multiplies and writes require that 'r' loop be innermost
				{
					for(size_t v=0; v<regC; v++)
					{
						for(size_t r=0; r<radix; r++)
						{

							std::string regRealIndex, regImagIndex;
							regRealIndex = linearRegs ? "(R" : regBaseCount;
							regImagIndex = linearRegs ? "(R" : regBaseCount;

							if(linearRegs)
							{
								RegBaseAndCountAndPos("", i*radix + r, regRealIndex); regRealIndex += "[0]).x";
								RegBaseAndCountAndPos("", i*radix + r, regImagIndex); regImagIndex += "[0]).y";
							}
							else
							{
								RegBaseAndCountAndPos("R", r, regRealIndex);
								RegBaseAndCountAndPos("I", r, regImagIndex);
							}

							if(regC != 1)
							{
								regRealIndex += ".s"; regRealIndex += SztToStr(v);
								regImagIndex += ".s"; regImagIndex += SztToStr(v);
							}


							if(flag == SR_TWMUL) // twiddle multiply operation
							{
								if(r == 0) // no twiddle muls needed
									continue;

								passStr += "\n\t{\n\t\t"; passStr += twType; passStr += " W = ";
								passStr += twTable; passStr += "["; passStr += SztToStr(algLS-1); passStr += " + ";
								passStr += SztToStr(radix-1); passStr += "*(("; passStr += SztToStr(numButterfly);
								passStr += "*me + "; passStr += SztToStr(butterflyIndex); passStr += ")%";
								passStr += SztToStr(algLS); passStr += ") + "; passStr += SztToStr(r-1);
								passStr += "];\n\t\t";
							}
							else	// 3-step twiddle
							{
								passStr += "\n\t{\n\t\t"; passStr += twType; passStr += " W = ";
								passStr += tw3StepFunc; passStr += "( ";
								passStr += "(("; passStr += SztToStr(numButterfly); passStr += "*me + ";
								passStr += SztToStr(butterflyIndex);
								passStr += ")%"; passStr += SztToStr(algLS); passStr += " + ";
								passStr += SztToStr(r*algLS); passStr += ") * b "; passStr += ");\n\t\t";
							}

							passStr += rType; passStr += " TR, TI;\n\t\t";
							if(fwd)
							{
								passStr += "TR = (W.x * "; passStr += regRealIndex; passStr += ") - (W.y * ";
								passStr += regImagIndex; passStr += ");\n\t\t";
								passStr += "TI = (W.y * "; passStr += regRealIndex; passStr += ") + (W.x * ";
								passStr += regImagIndex; passStr += ");\n\t\t";
							}
							else
							{
								passStr += "TR =  (W.x * "; passStr += regRealIndex; passStr += ") + (W.y * ";
								passStr += regImagIndex; passStr += ");\n\t\t";
								passStr += "TI = -(W.y * "; passStr += regRealIndex; passStr += ") + (W.x * ";
								passStr += regImagIndex; passStr += ");\n\t\t";
							}

							passStr += regRealIndex; passStr += " = TR;\n\t\t";
							passStr += regImagIndex; passStr += " = TI;\n\t}\n";

						}

						butterflyIndex++;
					}
				}
				else // write operation
				{
					for(size_t v=0; v<regC; v++)
					{
						for(size_t r=0; r<radix; r++)
						{
							for(size_t c=cStart; c<cEnd; c++) // component loop: 0 - real, 1 - imaginary
							{
								std::string tail;
								std::string regIndex;
								regIndex = linearRegs ? "(R" : regBaseCount;
								std::string buffer;

								// Write real & imag at once
								if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
								{
									assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag buffer strings are same for interleaved data
									buffer = bufferRe;
									RegBaseAndCountAndPos("", i*radix + r, regIndex); regIndex += "[0])";
									tail = "";
								}
								else
								{
									if(c == 0)
									{
										if(linearRegs) { RegBaseAndCountAndPos("", i*radix + r, regIndex); regIndex += "[0]).x"; }
										else		   { RegBaseAndCountAndPos("R", r, regIndex); }
										buffer = bufferRe;
										tail = interleaved ? ".x" : "";
									}
									else
									{
										if(linearRegs) { RegBaseAndCountAndPos("", i*radix + r, regIndex); regIndex += "[0]).y"; }
										else		   { RegBaseAndCountAndPos("I", r, regIndex); }
										buffer = bufferIm;
										tail = interleaved ? ".y" : "";
									}
								}

								if(regC != 1)
								{
									regIndex += ".s";
									regIndex += SztToStr(v);
								}

								passStr += "\n\t";
								passStr += buffer; passStr += "["; passStr += offset; passStr += " + ( ";

								if( (numButterfly * workGroupSize) > algLS )
								{
									passStr += "(("; passStr += SztToStr(numButterfly);
									passStr += "*me + "; passStr += SztToStr(butterflyIndex); passStr += ")/";
									passStr += SztToStr(algLS); passStr += ")*"; passStr += SztToStr(algL); passStr += " + (";
									passStr += SztToStr(numButterfly); passStr += "*me + "; passStr += SztToStr(butterflyIndex);
									passStr += ")%"; passStr += SztToStr(algLS); passStr += " + ";
								}
								else
								{
									passStr += SztToStr(numButterfly); passStr += "*me + "; passStr += SztToStr(butterflyIndex);
									passStr += " + ";
								}

								passStr += SztToStr(r*algLS); passStr += " )*"; passStr += SztToStr(stride); passStr += "]";
								passStr += tail; passStr += " = "; passStr += regIndex;
								if(scale != 1.0f) { passStr += " * "; passStr += FloatToStr(scale); passStr += FloatSuffix<PR>(); }
								passStr += ";";

								// Since we write real & imag at once, we break the loop
								if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
									break;
							}
						}

						butterflyIndex++;
					}

				}
			}

			assert(butterflyIndex <= numButterfly);
		}

		// Special SweepRegs function to carry out some R-C/C-R specific operations
		void SweepRegsRC(	size_t flag, bool fwd, bool interleaved, size_t stride, size_t component,
							double scale, bool setZero, bool batch2, bool oddt,
							const std::string &bufferRe, const std::string &bufferIm, const std::string &offset,
							std::string &passStr) const
		{
			assert( (flag == SR_READ ) ||
					(flag == SR_WRITE) );


			// component: 0 - real, 1 - imaginary, 2 - both
			size_t cStart, cEnd;
			switch(component)
			{
			case SR_COMP_REAL:	cStart = 0; cEnd = 1; break;
			case SR_COMP_IMAG:	cStart = 1; cEnd = 2; break;
			case SR_COMP_BOTH:	cStart = 0; cEnd = 2; break;
			default:	assert(false);
			}

			std::string rType  = RegBaseType<PR>(1);

			assert(r2c || c2r);
			assert(linearRegs);
			bool singlePass = ((position == 0) && (nextPass == NULL));

			size_t numCR = numButterfly * radix;
			if(!(numCR%2)) assert(!oddt);

			size_t rStart = 0;
			size_t rEnd = numCR;

			bool oddp = ((numCR%2) && (numCR > 1) && !setZero);
			if(oddp)
			{
				if(oddt)	{ rStart = numCR-1; rEnd = numCR+1; }
				else		{ rStart = 0;		rEnd = numCR-1; }
			}

			if(!oddp) assert(!oddt);

			for(size_t r=rStart; r<rEnd; r++)
			{
				for(size_t c=cStart; c<cEnd; c++) // component loop: 0 - real, 1 - imaginary
				{
					if(flag == SR_READ) // read operation
					{

						std::string tail, tail2;
						std::string regIndex = "(R";
						std::string buffer;

						if(c == 0)
						{
							RegBaseAndCountAndPos("", r, regIndex); regIndex += "[0]).x";
							buffer = bufferRe;
							tail  = interleaved ? ".x;" : ";";
							tail2 = interleaved ? ".y;" : ";";
						}
						else
						{
							RegBaseAndCountAndPos("", r, regIndex); regIndex += "[0]).y";
							buffer = bufferIm;
							tail  = interleaved ? ".y;" : ";";
							tail2 = interleaved ? ".x;" : ";";
						}


						size_t bid = numCR/2;
						bid = bid ? bid : 1;
						size_t cid, lid;

						if(oddt)
						{
							cid = r%2;
							lid = 1 + (numCR/2);
						}
						else
						{
							cid = r/bid;
							lid = 1 + r%bid;
						}

						std::string oddpadd = oddp ? " (me/2) + " : " ";

						std::string idxStr, idxStrRev;
						idxStr += SztToStr(bid); idxStr += "*me +"; idxStr += oddpadd; idxStr += SztToStr(lid);
						idxStrRev += SztToStr(length); idxStrRev += " - ("; idxStrRev += idxStr; idxStrRev += " )";

						bool act = ( fwd || ((cid == 0) && (!batch2)) || ((cid != 0) && batch2) );
						if(act)
						{
							passStr += "\n\t";
							passStr += regIndex;
							passStr += " = ";
						}

						if(setZero)
						{
							if(act) passStr += "0;";
						}
						else
						{
							if(act)
							{
								passStr += buffer;
								passStr += "["; passStr += offset; passStr += " + ( ";
							}

							if(fwd)
							{
								if(cid == 0)	passStr += idxStr;
								else			passStr += idxStrRev;
							}
							else
							{
								if(cid == 0)	{ if(!batch2) passStr += idxStr; }
								else			{ if(batch2)  passStr += idxStr; }
							}

							if(act)
							{
								passStr += " )*"; passStr += SztToStr(stride); passStr += "]";

								if(fwd) { passStr += tail; }
								else	{ if(!batch2) passStr += tail; else passStr += tail2; }
							}
						}
					}
					else // write operation
					{

						std::string tail;
						std::string regIndex = "(R";
						std::string regIndexPair = "(R";
						std::string buffer;

						// Write real & imag at once
						if(interleaved && (component == SR_COMP_BOTH))
						{
							assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag buffer strings are same for interleaved data
							buffer = bufferRe;
						}
						else
						{
							if(c == 0)
							{
								buffer = bufferRe;
								tail = interleaved ? ".x" : "";
							}
							else
							{
								buffer = bufferIm;
								tail = interleaved ? ".y" : "";
							}
						}


						size_t bid, cid, lid;
						if(singlePass && fwd)
						{
							bid = 1 + radix/2;
							lid = r;
							cid = r/bid;

							RegBaseAndCountAndPos("", r, regIndex); regIndex += "[0])";
							RegBaseAndCountAndPos("", (radix - r)%radix , regIndexPair); regIndexPair += "[0])";
						}
						else
						{
							bid = numCR/2;

							if(oddt)
							{
								cid = r%2;
								lid = 1 + (numCR/2);

								RegBaseAndCountAndPos("", r, regIndex); regIndex += "[0])";
								RegBaseAndCountAndPos("", r + 1, regIndexPair); regIndexPair += "[0])";
							}
							else
							{
								cid = r/bid;
								lid = 1 + r%bid;

								RegBaseAndCountAndPos("", r, regIndex); regIndex += "[0])";
								RegBaseAndCountAndPos("", r + bid, regIndexPair); regIndexPair += "[0])";
							}
						}


						if(!cid)
						{
							std::string oddpadd = oddp ? " (me/2) + " : " ";

							std::string sclStr = "";
							if(scale != 1.0f) { sclStr += " * "; sclStr += FloatToStr(scale); sclStr += FloatSuffix<PR>(); }

							if(fwd)
							{
								std::string idxStr, idxStrRev;
								idxStr += SztToStr(length/(2*workGroupSize)); idxStr += "*me +"; idxStr += oddpadd; idxStr += SztToStr(lid);
								idxStrRev += SztToStr(length); idxStrRev += " - ("; idxStrRev += idxStr; idxStrRev += " )";

								std::string val1Str, val2Str;

								val1Str += "\n\t";
								val1Str += buffer; val1Str += "["; val1Str += offset; val1Str += " + ( ";
								val1Str += idxStr; val1Str += " )*"; val1Str += SztToStr(stride); val1Str += "]";
								val1Str += tail; val1Str += " = ";

								val2Str += "\n\t";
								val2Str += buffer; val2Str += "["; val2Str += offset; val2Str += " + ( ";
								val2Str += idxStrRev; val2Str += " )*"; val2Str += SztToStr(stride); val2Str += "]";
								val2Str += tail; val2Str += " = ";

								std::string real1, imag1, real2, imag2;

								real1 +=  "("; real1 += regIndex; real1 += ".x + "; real1 += regIndexPair; real1 += ".x)*0.5";
								imag1 +=  "("; imag1 += regIndex; imag1 += ".y - "; imag1 += regIndexPair; imag1 += ".y)*0.5";
								real2 +=  "("; real2 += regIndex; real2 += ".y + "; real2 += regIndexPair; real2 += ".y)*0.5";
								imag2 += "(-"; imag2 += regIndex; imag2 += ".x + "; imag2 += regIndexPair; imag2 += ".x)*0.5";

								if(interleaved && (component == SR_COMP_BOTH))
								{
									val1Str += RegBaseType<PR>(2); val1Str += "( ";
									val2Str += RegBaseType<PR>(2); val2Str += "( ";

									if(!batch2) { val1Str += real1; val1Str += ", "; val1Str += "+"; val1Str += imag1;
												  val2Str += real1; val2Str += ", "; val2Str += "-"; val2Str += imag1; }
									else		{ val1Str += real2; val1Str += ", "; val1Str += "+"; val1Str += imag2;
												  val2Str += real2; val2Str += ", "; val2Str += "-"; val2Str += imag2; }

									val1Str += " )";
									val2Str += " )";
								}
								else
								{
									val1Str += " (";
									val2Str += " (";
									if(c == 0)
									{
										if(!batch2) { val1Str += real1;
													  val2Str += real1; }
										else		{ val1Str += real2;
													  val2Str += real2; }
									}
									else
									{
										if(!batch2) { val1Str += "+"; val1Str += imag1;
													  val2Str += "-"; val2Str += imag1; }
										else		{ val1Str += "+"; val1Str += imag2;
													  val2Str += "-"; val2Str += imag2; }
									}
									val1Str += " )";
									val2Str += " )";
								}

								val1Str += sclStr;
								val2Str += sclStr;

												passStr += val1Str; passStr += ";";
								if(rcFull)	{	passStr += val2Str; passStr += ";"; }
							}
							else
							{
								std::string idxStr, idxStrRev;
								idxStr += SztToStr(bid); idxStr += "*me +"; idxStr += oddpadd; idxStr += SztToStr(lid);
								idxStrRev += SztToStr(length); idxStrRev += " - ("; idxStrRev += idxStr; idxStrRev += " )";

								passStr += "\n\t";
								passStr += buffer; passStr += "["; passStr += offset; passStr += " + ( ";

								if(!batch2)	passStr += idxStr;
								else		passStr += idxStrRev;

								passStr += " )*"; passStr += SztToStr(stride); passStr += "]";
								passStr += tail; passStr += " = ";

								passStr += "( ";
								if(c == 0)
								{
									regIndex += ".x"; regIndexPair += ".x";

									if(!batch2)	{ passStr += regIndex; passStr += " - "; passStr += regIndexPair; }
									else		{ passStr += regIndex; passStr += " + "; passStr += regIndexPair; }
								}
								else
								{
									regIndex += ".y"; regIndexPair += ".y";

									if(!batch2)	{					passStr += regIndex; passStr += " + "; passStr += regIndexPair; }
									else		{ passStr += " - "; passStr += regIndex; passStr += " + "; passStr += regIndexPair; }
								}
								passStr += " )";
								passStr += sclStr;
								passStr += ";";
							}



							// Since we write real & imag at once, we break the loop
							if(interleaved && (component == SR_COMP_BOTH))
								break;
						}
					}
				}
			}

		}

		void CallButterfly(const std::string &bflyName, size_t regC, size_t numB, std::string &passStr) const
		{
			std::string regBase;
			RegBase(regC, regBase);

			for(size_t i=0; i<numB; i++)
			{
				std::string regBaseCount = regBase;
				RegBaseAndCount(i, regBaseCount);

				passStr += "\n\t";
				passStr += bflyName;
				passStr += "(";

				for(size_t r=0; ; r++)
				{
					if(linearRegs)
					{
						std::string regIndex = "R";
						RegBaseAndCountAndPos("", i*radix + r, regIndex);

						passStr += regIndex;
					}
					else
					{
						std::string regRealIndex(regBaseCount);
						std::string regImagIndex(regBaseCount);
						RegBaseAndCountAndPos("R", r, regRealIndex);
						RegBaseAndCountAndPos("I", r, regImagIndex);

						passStr += "&"; passStr += regRealIndex; passStr += ", ";
						passStr += "&"; passStr += regImagIndex;
					}

					if(r == radix-1)
					{
						passStr += ");";
						break;
					}
					else
					{
						passStr += ", ";
					}
				}
			}
		}
          public:
		Pass(	size_t positionVal, size_t lengthVal, size_t radixVal, size_t cnPerWIVal,
				size_t L, size_t LS, size_t R, bool linearRegsVal, bool r2cVal, bool c2rVal, bool rcFullVal, bool rcSimpleVal) :
			position(positionVal), length(lengthVal), radix(radixVal), cnPerWI(cnPerWIVal),
			algL(L), algLS(LS), algR(R), linearRegs(linearRegsVal),
			r2c(r2cVal), c2r(c2rVal), rcFull(rcFullVal), rcSimple(rcSimpleVal),
			enableGrouping(true),
			numB1(0), numB2(0), numB4(0),
			nextPass(NULL)
		{
			assert(radix <= length);
			assert(length%radix == 0);

			numButterfly = cnPerWI/radix;
			workGroupSize = length/cnPerWI;

			// Total number of butterflies (over all work-tems) must be divisible by LS
			assert( ((numButterfly*workGroupSize)%algLS) == 0 );

			// All butterflies in one work-item should always be part of no more than 1 FFT transform.
			// In other words, there should not be more than 1 FFT transform per work-item.
			assert(cnPerWI <= length);

			// Calculate the different types of Butterflies needed
			if(linearRegs || r2c || c2r)
			{
				numB1 = numButterfly;
			}
			else
			{
				numB4 = numButterfly/4;
				numB2 = (numButterfly%4)/2; // can be 0 or 1
				numB1 = (numButterfly%2); // can be 0 or 1

				assert(numButterfly == (numB4*4 + numB2*2 + numB1));
			}
		}

		size_t GetNumB1() const { return numB1; }
		size_t GetNumB2() const { return numB2; }
		size_t GetNumB4() const { return numB4; }

		size_t GetPosition() const { return position; }
		size_t GetRadix() const { return radix; }

		void SetNextPass(Pass<PR> *np) { nextPass = np; }
		void SetGrouping(bool grp) { enableGrouping = grp; }
     };
}

template<>
ampfftStatus FFTPlan::GetMax1DLengthPvt<Stockham> (size_t * longest) const
{
	// TODO  The caller has already acquired the lock on *this
	//	However, we shouldn't depend on it.

	//	Query the devices in this context for their local memory sizes
	//	How large a kernel we can generate depends on the *minimum* LDS
	//	size for all devices.
	//
	const FFTEnvelope * pEnvelope = NULL;
	this->GetEnvelope (& pEnvelope);
	BUG_CHECK (NULL != pEnvelope);

	ARG_CHECK (NULL != longest)
	size_t LdsperElement = this->ElementSize();
	size_t result = pEnvelope->limit_LocalMemSize /
		(1 * LdsperElement);
	result = FloorPo2 (result);
	*longest = result;
	return AMPFFT_SUCCESS;
}

