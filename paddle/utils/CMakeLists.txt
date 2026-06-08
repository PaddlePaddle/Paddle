add_subdirectory(string)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_library(
    pybind_util
    SRCS pybind.cc
    DEPS phi common)
endif()
cc_library(md5 SRCS md5.cc)
if(WITH_GPU)
  nv_library(
    md5_gpu
    SRCS md5_gpu.cu
    DEPS md5 xxhash)
elseif(WITH_ROCM)
  hip_library(
    md5_gpu
    SRCS md5_gpu.cu
    DEPS md5 xxhash)
endif()
