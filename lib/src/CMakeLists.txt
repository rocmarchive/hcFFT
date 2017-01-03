CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#Setting a Project name
SET(PROJECT_NAME "hcfft")
SET(PROJECT_NAME_EXT "hipfft")

PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")

# Build hcfft and hipfft from source on AMD platform
IF (${PLATFORM} MATCHES "hcc")
  # Find HCC compiler
  FIND_PACKAGE(HC++ 1.0 REQUIRED)

  # Find HIP
  set (HIP_PATH $ENV{HIP_PATH})
  if (NOT DEFINED HIP_PATH)
  	set (HIP_PATH /opt/rocm/hip)
  endif()

  execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)
  MESSAGE ("HIP_PATH=" ${HIP_PATH})

  #Setting a variable for source files
  FILE(GLOB HCFFTSRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  SET(HIP_SHARED_OBJ $ENV{HIP_SHARED_OBJ})

  execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG} --install --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)

  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${HIP_PATH}/include -D__HIP_PLATFORM_HCC__= -DHIP_SHARED_OBJ=")
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HIP_PATH}/lib -lhip_hcc")

  # Setting header dependencies for hcfft library generation
  FOREACH(src_file ${HCFFTSRCS})
    SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  ENDFOREACH()

  #Generating hcfft shared object
  ADD_LIBRARY("${PROJECT_NAME}" SHARED  ${HCFFTSRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME}" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME}" hc_am)

  INSTALL(TARGETS "${PROJECT_NAME}" 
   RUNTIME DESTINATION lib
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib
  )

  file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
  INSTALL(FILES ${headers} DESTINATION include)

  IF(DEFINED HIP_SHARED_OBJ)
    SET(HIPFFTSRCS ${HCFFTSRCS} ${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipfft.cpp)

    # Setting header dependencies for hipfft library generation
    FOREACH(src_file ${HIPFFTSRCS})
     SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
    ENDFOREACH()

    #Generating hipfft shared object
    ADD_LIBRARY("${PROJECT_NAME_EXT}" SHARED ${HIPFFTSRCS})
    SET_PROPERTY(TARGET "${PROJECT_NAME_EXT}" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
    TARGET_LINK_LIBRARIES("${PROJECT_NAME_EXT}" hc_am)

    INSTALL(TARGETS "${PROJECT_NAME_EXT}" 
      RUNTIME DESTINATION lib
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
    )

    file(GLOB hcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/hcc_detail/*.h")
    file(GLOB nvcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvcc_detail/*.h")
    INSTALL(FILES ${hcc_detail} DESTINATION include/hcc_detail)
    INSTALL(FILES ${nvcc_detail} DESTINATION include/nvcc_detail)
  ENDIF()
ELSEIF (${PLATFORM} MATCHES "nvcc")
  MESSAGE(STATUS "Building hipfft.cpp")
  SET(HIPFFTSRCS ${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipfft.cpp)
  SET(HIP_SHARED_OBJ $ENV{HIP_SHARED_OBJ})
 
 ADD_DEFINITIONS(-DHIP_SHARED_OBJ)
  set (CXXFLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${HIP_PATH}/include -I/usr/local/cuda/include -DHIP_SHARED_OBJ= -D__HIP_PLATFORM_NVCC__=")
  set (LDFLAGS "-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib")
  FOREACH(src_file ${HIPFFTSRCS})
	  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CXXFLAGS} ")
  ENDFOREACH()
  ADD_LIBRARY("${PROJECT_NAME_EXT}" SHARED ${HIPFFTSRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME_EXT}" APPEND_STRING PROPERTY LINK_FLAGS " ${LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME_EXT}" cudart cufft)
  INSTALL(TARGETS "${PROJECT_NAME_EXT}" 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
  file(GLOB hcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/hcc_detail/*.h")
  file(GLOB nvcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvcc_detail/*.h")
  INSTALL(FILES ${hcc_detail} DESTINATION include/hcc_detail)
  INSTALL(FILES ${nvcc_detail} DESTINATION include/nvcc_detail)
ENDIF()



