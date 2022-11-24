if(NOT APPLE)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")
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
if(WITH_OMP)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
endif()

if(NOT (CUDA_VERSION GREATER 10.2))
  set(CUDA_NVCC_FLAGS
      "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
  set(CUDA_NVCC_FLAGS
      "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
endif()

if(CUDA_VERSION GREATER 7.6)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
endif()

if(CUDA_VERSION GREATER 8.9)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
endif()

if(CUDA_VERSION GREATER 9.9)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
endif()

if(CUDA_VERSION GREATER 11.0)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
endif()

if(NOT APPLE)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++14")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
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
  message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_time.cu")
  message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu.cu")
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

# cc_test(
#   test_cpu
#   SRCS tests/test_cpu.cpp
#   DEPS warprnnt)
# cc_test(
#   test_time
#   SRCS tests/test_time.cpp
#   DEPS warprnnt)

# if(WITH_GPU)
#   # cuda_add_executable(test_time_gpu tests/test_time.cu tests/random.cpp)
#   # target_link_libraries(test_time_gpu warprnnt ${CUDA_curand_LIBRARY})
#   # set_target_properties(
#   #  test_time_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14")

#   # cuda_add_executable(test_gpu tests/test_gpu.cu tests/random.cpp)
#   # target_link_libraries(test_gpu warprnnt ${CUDA_curand_LIBRARY})
#   # set_target_properties(test_gpu PROPERTIES COMPILE_FLAGS
#   #                                          "${CMAKE_CXX_FLAGS} --std=c++14")
# endif()

# add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp)
# target_link_libraries(test_cpu warprnnt)
# set_target_properties(test_cpu PROPERTIES COMPILE_FLAGS
#                                          "${CMAKE_CXX_FLAGS} --std=c++14")

# add_executable(test_time tests/test_time.cpp tests/random.cpp)
# target_link_libraries(test_time warprnnt)
# set_target_properties(test_time PROPERTIES COMPILE_FLAGS
#                                           "${CMAKE_CXX_FLAGS} --std=c++14")

# install(
#   TARGETS warprnnt
#   RUNTIME DESTINATION "bin"
#   LIBRARY DESTINATION "lib"
#   ARCHIVE DESTINATION "lib")

# install(FILES include/rnnt.h DESTINATION "include")
