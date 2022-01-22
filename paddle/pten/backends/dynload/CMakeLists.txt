cc_library(pten_dynamic_loader SRCS dynamic_loader.cc DEPS enforce glog gflags)

list(APPEND CUDA_SRCS cublas.cc cublasLt.cc cudnn.cc curand.cc cusolver.cc cusparse.cc  nvtx.cc cufft.cc)

if (NOT WITH_NV_JETSON)
  list(APPEND CUDA_SRCS nvjpeg.cc)
endif()

if (WITH_ROCM)
  list(APPEND HIP_SRCS rocblas.cc miopen.cc hiprand.cc hipfft.cc)
endif()

# There is no macOS version of NCCL.
# Disable nvrtc and cuda_driver api on MacOS, and only do a early test on Linux and Windows.
if (NOT APPLE)
  list(APPEND CUDA_SRCS nvrtc.cc cuda_driver.cc)
  if (WITH_NCCL)
    list(APPEND CUDA_SRCS nccl.cc)
  endif()
  if (WITH_ROCM)
    list(APPEND HIP_SRCS hiprtc.cc rocm_driver.cc)
    if (WITH_RCCL)
      list(APPEND HIP_SRCS rccl.cc)
    endif()
  endif()
endif()

if (TENSORRT_FOUND)
  list(APPEND CUDA_SRCS tensorrt.cc)
endif()

configure_file(cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
if (CUPTI_FOUND)
  list(APPEND CUDA_SRCS cupti.cc)
endif(CUPTI_FOUND)
if(WITH_ROCM)
  hip_library(pten_dynload_cuda SRCS ${HIP_SRCS} DEPS pten_dynamic_loader)
  cc_library(pten_dynload_warpctc SRCS warpctc.cc DEPS pten_dynamic_loader warpctc)
elseif (WITH_ASCEND_CL)
  cc_library(pten_dynload_warpctc SRCS warpctc.cc DEPS pten_dynamic_loader warpctc npu_hccl)
else()
  nv_library(pten_dynload_cuda SRCS ${CUDA_SRCS} DEPS pten_dynamic_loader)
  cc_library(pten_dynload_warpctc SRCS warpctc.cc DEPS pten_dynamic_loader warpctc)
endif()
if (WITH_MKLML)
  cc_library(pten_dynload_mklml SRCS mklml.cc DEPS pten_dynamic_loader mklml)
endif()

cc_library(pten_dynload_lapack SRCS lapack.cc DEPS pten_dynamic_loader)
add_dependencies(pten_dynload_lapack extern_lapack)
# TODO(TJ): add iomp, mkldnn?

if (MKL_FOUND AND WITH_ONEMKL)
  message("ONEMKL INCLUDE directory is ${MKL_INCLUDE}")
  cc_library(pten_dynload_mklrt SRCS mklrt.cc DEPS pten_dynamic_loader)
  target_include_directories(pten_dynload_mklrt PRIVATE ${MKL_INCLUDE})
endif()
