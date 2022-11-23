if(APPLE)
  cmake_minimum_required(VERSION 3.4)
else()
  cmake_minimum_required(VERSION 2.8)
endif()

project(rnnt_release)

if(NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")
endif()

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
  add_definitions(-DAPPLE)
endif()

include_directories(include)

find_package(CUDA)
message(STATUS "cuda found ${CUDA_FOUND}")

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

option(WITH_GPU "compile warp-rnnt with cuda." ${CUDA_FOUND})
option(WITH_OMP "compile warp-rnnt with openmp." ON)

if(NOT WITH_OMP)
  add_definitions(-DRNNT_DISABLE_OMP)
endif()
if(WITH_OMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
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
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
endif()

if(APPLE)
  exec_program(
    uname ARGS
    -v
    OUTPUT_VARIABLE DARWIN_VERSION)
  string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  message(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

  #for el capitain have to use rpath

  if(DARWIN_VERSION LESS 15)
    set(CMAKE_SKIP_RPATH TRUE)
  endif()

else()
  #always skip for linux
  set(CMAKE_SKIP_RPATH TRUE)
endif()

if(WITH_GPU)

  message(STATUS "Building shared library with GPU support")

  cuda_add_library(warprnnt SHARED src/rnnt_entrypoint.cu)
  if(!Torch_FOUND)
    target_link_libraries(warprnnt ${CUDA_curand_LIBRARY})
  endif()

  cuda_add_executable(test_time_gpu tests/test_time.cu tests/random.cpp)
  target_link_libraries(test_time_gpu warprnnt ${CUDA_curand_LIBRARY})
  set_target_properties(
    test_time_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

  cuda_add_executable(test_gpu tests/test_gpu.cu tests/random.cpp)
  target_link_libraries(test_gpu warprnnt ${CUDA_curand_LIBRARY})
  set_target_properties(test_gpu PROPERTIES COMPILE_FLAGS
                                            "${CMAKE_CXX_FLAGS} --std=c++11")

else()
  message(STATUS "Building shared library with no GPU support")

  if(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
  endif()

  add_library(warprnnt SHARED src/rnnt_entrypoint.cpp)

endif()

add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp)
target_link_libraries(test_cpu warprnnt)
set_target_properties(test_cpu PROPERTIES COMPILE_FLAGS
                                          "${CMAKE_CXX_FLAGS} --std=c++11")

add_executable(test_time tests/test_time.cpp tests/random.cpp)
target_link_libraries(test_time warprnnt)
set_target_properties(test_time PROPERTIES COMPILE_FLAGS
                                           "${CMAKE_CXX_FLAGS} --std=c++11")

install(
  TARGETS warprnnt
  RUNTIME DESTINATION "bin"
  LIBRARY DESTINATION "lib"
  ARCHIVE DESTINATION "lib")

install(FILES include/rnnt.h DESTINATION "include")
