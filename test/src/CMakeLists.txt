# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

#Setting a variable for source files
SET (TESTSRCS
    fft_test.cpp
    fft_timer_test.cpp
    )

IF(WIN32)
  #include the additional libraries 
  #settings->c/c++->general->additional directories 
  SET(TESTSRCS ${TESTSRCS})
  ADD_DEFINITIONS(-D__LITTLE_ENDIAN__)
  ADD_DEFINITIONS(-DTARGET_OS_WIN32=1)
    
  IF(MSVC)
    ADD_DEFINITIONS(/W3)
    SET (DEBUG_PATH "../../Build/vc11-x86_64/Debug/")
    SET (RELEASE_PATH "../../Build/vc11-x86_64/Release/")
    include_directories( ../../source/)
    SET(CMAKE_CXX_STANDARD_LIBRARIES ${CMAKE_CXX_STANDARD_LIBRARIES} libhcfft.lib)
    IF(EXISTS "${DEBUG_PATH}/libhcfft.lib")
     LINK_DIRECTORIES (${DEBUG_PATH})
    ELSE(EXISTS "${DEBUG_PATH}/libhcfft.lib")
     LINK_DIRECTORIES (${RELEASE_PATH})
    ENDIF(EXISTS "${DEBUG_PATH}/libhcfft.lib")
    STRING( REPLACE ".cpp" "" testname ${TESTSRCS} )
    ADD_EXECUTABLE( ${testname} ${TESTSRCS})
  ELSE(MSVC)
    MESSAGE(FATAL_ERROR "TARGET IDE ISNT VISUAL STUDIO 11/12.... EXITING")
  ENDIF(MSVC)
ELSE(WIN32)
  execute_process(COMMAND ${HCC_CONFIG}  --cxxflags
                          OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG} --ldflags
                          OUTPUT_VARIABLE HCC_LDFLAGS)
  SET(HCFFT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/")
  SET(HCFFT_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib/src")

  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${HCFFT_INCLUDE_PATH}")
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HCFFT_LIBRARY_PATH}")
  SET (LINK "-lhcfft -lhc_am")

  FOREACH(test_file ${TESTSRCS})
    SET_PROPERTY(SOURCE ${test_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS}")
    STRING( REPLACE "_test.cpp" "" testname ${test_file} )
    ADD_EXECUTABLE(${testname} ${test_file} )
    SET_PROPERTY(TARGET ${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ${LINK}")
  ENDFOREACH()

ENDIF(WIN32)
