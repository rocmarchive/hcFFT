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

