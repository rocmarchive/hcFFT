# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

#Setting a Project name
SET(PROJECT_NAME "libhcffttest")
PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

#Setting a variable for source files
SET (TESTSRCS
    fft_test.cpp  
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
  # Choice to take compilation flags from source or package
  if(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${CLAMP_CONFIG} --build --cxxflags
                            OUTPUT_VARIABLE CLAMP_CXXFLAGS)
    execute_process(COMMAND ${CLAMP_CONFIG} --build --ldflags --shared
                            OUTPUT_VARIABLE CLAMP_LDFLAGS)
  else(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${CLAMP_CONFIG} --cxxflags
                            OUTPUT_VARIABLE CLAMP_CXXFLAGS)
    execute_process(COMMAND ${CLAMP_CONFIG} --ldflags --shared
                            OUTPUT_VARIABLE CLAMP_LDFLAGS)
  endif(EXISTS ${MCWHCCBUILD})


  string(STRIP "${CLAMP_CXXFLAGS}" CLAMP_CXXFLAGS)
  string(STRIP "${CLAMP_LDFLAGS}" CLAMP_LDFLAGS)
  set (CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS} -Wall -Wno-deprecated-register -Wdeprecated-declarations")
  set (CLAMP_LDFLAGS "${CLAMP_LDFLAGS}")


  SET (LINK "-lhcfft")
  SET_PROPERTY(SOURCE ${TESTSRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
  STRING( REPLACE "_test.cpp" "" testname ${TESTSRCS} )
  ADD_EXECUTABLE( ${testname} ${TESTSRCS} )
  SET_PROPERTY(TARGET ${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${LINK}")

ENDIF(WIN32)
