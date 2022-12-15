if(APPLE)
  add_definitions(-DAPPLE)
endif()

include_directories(include)

option(USE_NAIVE_KERNEL "use naive alpha-beta kernel" OFF)
option(DEBUG_TIME "output kernel time" OFF)
option(DEBUG_KERNEL "output alpha beta" OFF)
if(USE_NAIVE_KERNEL)
  add_definitions(-DUSE_NAIVE_KERNEL)
endif()
if(DEBUG_TIME)
  add_definitions(-DDEBUG_TIME)
endif()
if(DEBUG_KERNEL)
  add_definitions(-DDEBUG_KERNEL)
endif()

if(NOT WITH_MKL)
  message(STATUS "RNNT-Loss w/o OMP")
  add_definitions(-DRNNT_DISABLE_OMP)
endif()

# need to be at least 30 or __shfl_down in reduce wont compile
if(CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
endif()

# sm35 is deprecated after cuda 12.0
if(CUDA_VERSION VERSION_LESS "12.0")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

if(CUDA_VERSION VERSION_GREATER "7.6")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
endif()

if((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
endif()

if((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
endif()

if((CUDA_VERSION VERSION_GREATER "11.0") OR (CUDA_VERSION VERSION_EQUAL "11.0"))
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
endif()

if((CUDA_VERSION VERSION_GREATER "11.2") OR (CUDA_VERSION VERSION_EQUAL "11.2"))
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
endif()

if((CUDA_VERSION VERSION_GREATER "11.8") OR (CUDA_VERSION VERSION_EQUAL "11.8"))
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_90,code=sm_90")
endif()

message(STATUS "CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")

file(GLOB RNNT_CC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB RNNT_CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
message(STATUS "RNNT_CC_SOURCES=${RNNT_CC_SOURCES}")
message(STATUS "RNNT_CU_SOURCES=${RNNT_CU_SOURCES}")

if(WITH_GPU)
  nv_library(
    warprnnt
    SRCS ${RNNT_CU_SOURCES}
    DEPS dynload_cuda)
elseif(WITH_ROCM)
  hip_library(
    warprnnt
    SRCS ${RNNT_CU_SOURCES}
    DEPS dynload_cuda)
else()
  cc_library(warprnnt SRCS ${RNNT_CC_SOURCES})
endif()

if(WITH_GPU)
  nv_test(
    test_rnnt_time_gpu
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_time.cu"
    DEPS warprnnt)
  nv_test(
    test_rnnt_gpu
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu.cu"
    DEPS warprnnt)
elseif(WITH_ROCM)
  hip_test(
    test_rnnt_time_gpu
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_time.cu"
    DEPS warprnnt)
  hip_test(
    test_rnnt_gpu
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu.cu"
    DEPS warprnnt)
elseif(WITH_XPU)
  message(STATUS "warprnnt test not support xpu.")
else()
  message(STATUS "warprnnt default branch")
  message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu.cpp")
  message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_time.cpp")
  cc_test(
    test_rnnt_cpu
    SRCS "tests/test_cpu.cpp"
    DEPS warprnnt)
  cc_test(
    test_rnnt_time_cpu
    SRCS "tests/test_time.cpp"
    DEPS warprnnt)
endif()
