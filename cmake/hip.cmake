if(NOT WITH_AMD_GPU)
    return()
endif()

include_directories("/opt/rocm/include")
include_directories("/opt/rocm/hipblas/include")
include_directories("/opt/rocm/hiprand/include")
include_directories("/opt/rocm/rocrand/include")
include_directories("/opt/rocm/rccl/include")
include_directories("/opt/rocm/thrust")

list(APPEND EXTERNAL_LIBS "-L/opt/rocm/lib/ -lhip_hcc")

set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -fPIC -DPADDLE_WITH_HIP -std=c++14" )

if(WITH_DSO)
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -DPADDLE_USE_DSO")
endif(WITH_DSO)

if(WITH_DOUBLE)
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -DPADDLE_TYPE_DOUBLE")
endif(WITH_DOUBLE)

if(WITH_TESTING)
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -DPADDLE_WITH_TESTING")
endif(WITH_TESTING)

if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND HIP_HCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE  STREQUAL "RelWithDebInfo")
    list(APPEND HIP_HCC_FLAGS  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(CMAKE_BUILD_TYPE  STREQUAL "MinSizeRel")
    list(APPEND HIP_HCC_FLAGS  ${CMAKE_CXX_FLAGS_MINSIZEREL})
endif()

if("x${HCC_HOME}" STREQUAL "x")
  set(HCC_HOME "/opt/rocm/hcc")
endif()

set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -shared")
set(CMAKE_HIP_CREATE_SHARED_MODULE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -shared")

