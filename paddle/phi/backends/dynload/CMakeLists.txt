cc_library(
  phi_dynamic_loader
  SRCS dynamic_loader.cc port.cc
  DEPS enforce glog gflags)

list(
  APPEND
  CUDA_SRCS
  cublas.cc
  cublasLt.cc
  cudnn.cc
  curand.cc
  cusolver.cc
  cusparse.cc
  nvtx.cc
  cufft.cc)

if(NOT WITH_NV_JETSON)
  list(APPEND CUDA_SRCS nvjpeg.cc)
endif()

if(WITH_ROCM)
  list(
    APPEND
    HIP_SRCS
    rocblas.cc
    miopen.cc
    hiprand.cc
    hipfft.cc
    rocsparse.cc)
endif()

# There is no macOS version of NCCL.
# Disable nvrtc and cuda_driver api on MacOS, and only do a early test on Linux and Windows.
if(NOT APPLE)
  list(APPEND CUDA_SRCS nvrtc.cc cuda_driver.cc)
  if(WITH_NCCL)
    list(APPEND CUDA_SRCS nccl.cc)
  endif()
  if(WITH_ROCM)
    list(APPEND HIP_SRCS hiprtc.cc rocm_driver.cc)
    if(WITH_RCCL)
      list(APPEND HIP_SRCS rccl.cc)
    endif()
    if(CUPTI_FOUND)
      list(APPEND HIP_SRCS cupti.cc)
    endif()
  endif()
endif()

if(TENSORRT_FOUND)
  list(APPEND CUDA_SRCS tensorrt.cc)
endif()

if(CUSPARSELT_FOUND)
  list(APPEND CUDA_SRCS cusparseLt.cc)
endif()

configure_file(cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
if(CUPTI_FOUND)
  list(APPEND CUDA_SRCS cupti.cc)
endif()
if(WITH_ROCM)
  hip_library(
    phi_dynload_cuda
    SRCS ${HIP_SRCS}
    DEPS phi_dynamic_loader)
  cc_library(
    phi_dynload_warpctc
    SRCS warpctc.cc
    DEPS phi_dynamic_loader warpctc)
  cc_library(
    phi_dynload_warprnnt
    SRCS warprnnt.cc
    DEPS phi_dynamic_loader warprnnt)
else()
  nv_library(
    phi_dynload_cuda
    SRCS ${CUDA_SRCS}
    DEPS phi_dynamic_loader)
  cc_library(
    phi_dynload_warpctc
    SRCS warpctc.cc
    DEPS phi_dynamic_loader warpctc)
  cc_library(
    phi_dynload_warprnnt
    SRCS warprnnt.cc
    DEPS phi_dynamic_loader warprnnt)
endif()
if(WITH_MKLML)
  cc_library(
    phi_dynload_mklml
    SRCS mklml.cc
    DEPS phi_dynamic_loader mklml)
endif()

if(WITH_FLASHATTN)
  cc_library(
    phi_dynload_flashattn
    SRCS flashattn.cc
    DEPS phi_dynamic_loader flashattn)
endif()

cc_library(
  phi_dynload_lapack
  SRCS lapack.cc
  DEPS phi_dynamic_loader)
add_dependencies(phi_dynload_lapack extern_lapack)
# TODO(TJ): add iomp, mkldnn?

if(MKL_FOUND AND WITH_ONEMKL)
  message("ONEMKL INCLUDE directory is ${MKL_INCLUDE}")
  cc_library(
    phi_dynload_mklrt
    SRCS mklrt.cc
    DEPS phi_dynamic_loader)
  target_include_directories(phi_dynload_mklrt PRIVATE ${MKL_INCLUDE})
endif()

if(WITH_CUDNN_FRONTEND)
  nv_test(
    cudnn_frontend_test
    SRCS cudnn_frontend_test.cc
    DEPS phi_dynload_cuda cudnn-frontend)
endif()
