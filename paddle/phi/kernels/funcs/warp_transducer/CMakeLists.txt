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

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
   OR WIN32)
  set(WITH_OMP OFF)
else()
  set(WITH_OMP ON)
endif()

if(NOT WITH_OMP)
  add_definitions(-DRNNT_DISABLE_OMP)
endif()

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
else()
  cc_test(
    test_rnnt_cpu
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu.cpp"
    DEPS warprnnt)
  cc_test(
    test_rnnt_time
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_time.cpp"
    DEPS warprnnt)
endif()
