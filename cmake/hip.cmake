if(NOT WITH_ROCM_PLATFORM)
    return()
endif()

include_directories("${ROCM_PATH}/include")
include_directories("${ROCM_PATH}/hip/include")
include_directories("${ROCM_PATH}/miopen/include")
include_directories("${ROCM_PATH}/hipblas/include")
include_directories("${ROCM_PATH}/rocblas/include")
include_directories("${ROCM_PATH}/hiprand/include")
include_directories("${ROCM_PATH}/rocrand/include")
include_directories("${ROCM_PATH}/rccl/include")

include_directories("${ROCM_PATH}/rocthrust/include/")
include_directories("${ROCM_PATH}/hipcub/include/")
include_directories("${ROCM_PATH}/rocprim/include/")
include_directories("${ROCM_PATH}/hipsparse/include/")
include_directories("${ROCM_PATH}/rocsparse/include/")
include_directories("${ROCM_PATH}/rocfft/include/")

set(HIP_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS "")
set(HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS "")
# now default is clang
set(HIP_COMPILER "clang")

list(APPEND EXTERNAL_LIBS "-L${ROCM_PATH}/lib/ -lhip_hcc")
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -fPIC -DPADDLE_WITH_HIP -DEIGEN_USE_HIP -DEIGEN_USE_GPU -D__HIP_NO_HALF_CONVERSIONS__ -std=c++11 --amdgpu-target=gfx906" )

if(WITH_RCCL)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_RCCL")
endif()

if(NOT WITH_PYTHON)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_NO_PYTHON")
endif(NOT WITH_PYTHON)

if(WITH_DSO)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_USE_DSO")
endif(WITH_DSO)

if(WITH_TESTING)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_TESTING")
endif(WITH_TESTING)

if(WITH_DISTRIBUTE)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_DISTRIBUTE")
endif(WITH_DISTRIBUTE)

if(WITH_GRPC)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_GRPC")
endif(WITH_GRPC)

if(WITH_MKLDNN)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DPADDLE_WITH_MKLDNN")
endif(WITH_MKLDNN)

set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -DANY_IMPL_ANY_CAST_MOVEABLE")

if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE  STREQUAL "RelWithDebInfo")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(CMAKE_BUILD_TYPE  STREQUAL "MinSizeRel")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_MINSIZEREL})
endif()

if("${HIP_COMPILER}" STREQUAL "hcc")
    if("x${HCC_HOME}" STREQUAL "x")
      set(HCC_HOME "${ROCM_PATH}/hcc")
    endif()

    set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -ldl --amdgpu-target=gfx906 ")
    set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -shared --amdgpu-target=gfx906")
    set(CMAKE_HIP_CREATE_SHARED_MODULE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -shared --amdgpu-target=gfx906")

elseif("${HIP_COMPILER}" STREQUAL "clang")
    
    if("x${HIP_CLANG_PATH}" STREQUAL "x")
        set(HIP_CLANG_PATH  "${ROCM_PATH}/llvm/bin")
    endif()

    #Number of parallel jobs by default is 1
    if(NOT DEFINED HIP_CLANG_NUM_PARALLEL_JOBS)
      set(HIP_CLANG_NUM_PARALLEL_JOBS 1)
    endif()
    #Add support for parallel build and link
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      check_cxx_compiler_flag("-parallel-jobs=1" HIP_CLANG_SUPPORTS_PARALLEL_JOBS)
    endif()
    if(HIP_CLANG_NUM_PARALLEL_JOBS GREATER 1)
      if(${HIP_CLANG_SUPPORTS_PARALLEL_JOBS})
        set(HIP_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS "-parallel-jobs=${HIP_CLANG_NUM_PARALLEL_JOBS} -Wno-format-nonliteral")
        set(HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS "-parallel-jobs=${HIP_CLANG_NUM_PARALLEL_JOBS}")
      else()
        message("clang compiler doesn't support parallel jobs")
      endif()
    endif()


    # Set the CMake Flags to use the HIP-Clang Compiler.
    set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> --amdgpu-target=gfx906")
    set(CMAKE_HIP_CREATE_SHARED_MODULE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <LINK_LIBRARIES> -shared --amdgpu-target=gfx906" )
    set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -ldl --amdgpu-target=gfx906")
endif()
