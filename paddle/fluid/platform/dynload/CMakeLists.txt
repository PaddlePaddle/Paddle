cc_library(dynamic_loader SRCS dynamic_loader.cc DEPS glog gflags enforce)

list(APPEND CUDA_SRCS cublas.cc cudnn.cc curand.cc)

# There is no macOS version of NCCL.
if (NOT APPLE AND NOT WIN32)
  list(APPEND CUDA_SRCS nccl.cc)
endif()

if (TENSORRT_FOUND)
  list(APPEND CUDA_SRCS tensorrt.cc)
endif()

configure_file(cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
if (CUPTI_FOUND)
    list(APPEND CUDA_SRCS cupti.cc)
endif(CUPTI_FOUND)
nv_library(dynload_cuda SRCS ${CUDA_SRCS} DEPS dynamic_loader)
list(APPEND HIP_SRCS hipblas.cc miopen.cc hiprand.cc rccl.cc)
hip_library(dynload_hip SRCS ${HIP_SRCS} DEPS dynamic_loader)
if (NOT WIN32)
cc_library(dynload_warpctc SRCS warpctc.cc DEPS dynamic_loader warpctc)
endif(NOT WIN32)
if (WITH_MKLML)
    cc_library(dynload_mklml SRCS mklml.cc DEPS dynamic_loader mklml)
endif()
# TODO(TJ): add iomp, mkldnn?
