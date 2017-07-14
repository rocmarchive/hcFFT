# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

SET (HCSRCS hcfft_Create_destroy_Plan.cpp)

execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                          OUTPUT_VARIABLE HCC_CXXFLAGS)
execute_process(COMMAND ${HCC_CONFIG}  --install --ldflags
                          OUTPUT_VARIABLE HCC_LDFLAGS)

SET(HCFFT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/")
SET(HCFFT_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/src")

string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${HCFFT_INCLUDE_PATH}")
string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HCFFT_LIBRARY_PATH}")
SET (LINK "-lhcfft -lhc_am")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
FOREACH(test_file ${HCSRCS})
  SET_PROPERTY(SOURCE ${test_file} ../gtest-all.cpp ../gtest_main.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} -DGTEST_HAS_TR1_TUPLE=0 ")
  get_filename_component (name_without_extension ${test_file} NAME_WE)
  ADD_EXECUTABLE(${name_without_extension} ${test_file} ../gtest-all.cpp ../gtest_main.cpp)
  SET_PROPERTY(TARGET ${name_without_extension} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ${LINK}")
ENDFOREACH()
