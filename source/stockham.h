#include <iostream>
#include <cassert>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include "ampfftlib.h"

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

        inline std::string ampHeader()
	{
                return "#include \"ampfftlib.h\"\n"
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
}
