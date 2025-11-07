set(DYNLOAD_COMMON_SRCS dynamic_loader.cc warpctc.cc warprnnt.cc lapack.cc)
if(WITH_ASCEND_CL)
  list(REMOVE_ITEM DYNLOAD_COMMON_SRCS warprnnt.cc)
endif()
if(WIN32)
  list(
    APPEND
    CUDA_SRCS
    cublasLt.cc
    curand.cc
    cusolver.cc
    cusparse.cc
    nvtx.cc
    cufft.cc
    cutlass_conv2d.cc
    cutlass_gemm_epilogue.cc)
  nv_library(
    dynload_cudnn static
    SRCS cudnn.cc
    DEPS dynload_common)
  nv_library(
    dynload_cublas static
    SRCS cublas.cc
    DEPS dynload_common)
else()
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
    cufft.cc
    cutlass_conv2d.cc
    cutlass_gemm_epilogue.cc)
endif()

if(NOT WITH_NV_JETSON)
  list(APPEND CUDA_SRCS nvjpeg.cc)
endif()

if(WITH_ROCM)
  list(
    APPEND
    HIP_SRCS
    rocblas.cc
    hipblasLt.cc
    miopen.cc
    hiprand.cc
    hipfft.cc
    rocsparse.cc
    rocsolver.cc)
endif()

if(WITH_XPU_FFT)
  list(APPEND XPU_SRCS xpufft.cc)
endif()

# There is no macOS version of NCCL.
# Disable nvrtc and cuda_driver api on macOS, and only do an early test on Linux and Windows.
if(NOT APPLE)
  if(WIN32)
    list(APPEND CUDA_SRCS nvrtc.cc)
    if(WITH_GPU)
      nv_library(
        dynload_cuda static
        SRCS cuda_driver.cc
        DEPS dynload_common)
    endif()
  else()
    list(APPEND CUDA_SRCS nvrtc.cc cuda_driver.cc)
  endif()
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
  if(WIN32)
    if(WITH_GPU)
      nv_library(
        dynload_tensorrt static
        SRCS tensorrt.cc
        DEPS)
    endif()
  else()
    list(APPEND CUDA_SRCS tensorrt.cc)
  endif()
endif()

if(CUSPARSELT_FOUND)
  list(APPEND CUDA_SRCS cusparseLt.cc)
endif()

configure_file(cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
if(CUPTI_FOUND)
  list(APPEND CUDA_SRCS cupti.cc)
endif()

if(WITH_MKLML)
  # Only deps libmklml.so, not link
  add_library(dynload_mklml STATIC mklml.cc)
  add_dependencies(dynload_mklml mklml)
  if(WIN32)
    target_link_libraries(dynload_mklml ${MKLML_IOMP_LIB})
  else()
    target_link_libraries(dynload_mklml
                          "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
  endif()
endif()

if(WITH_HML)
  # Only deps libhml_rt.so, not link
  add_library(dynload_hml STATIC hml.cc)
  add_dependencies(dynload_hml hml)
  if(WIN32)
    message(FATAL_ERROR "HML is not supported on Windows")
  else()
    target_link_libraries(dynload_hml "-L${HML_LIB_DIR} -lgomp -Wl,--as-needed")
  endif()
endif()

if(WITH_XPU)
  if(CUPTI_FOUND)
    collect_srcs(backends_srcs SRCS xpti.cc nvtx.cc cupti.cc)
  else()
    collect_srcs(backends_srcs SRCS xpti.cc nvtx.cc)
  endif()
endif()

if(WITH_FLAGCX)
  collect_srcs(backends_srcs SRCS flagcx.cc)
endif()

if(WITH_FLASHATTN)
  list(APPEND DYNLOAD_COMMON_SRCS flashattn.cc)
endif()

if(WITH_FLASHATTN_V3)
  list(APPEND DYNLOAD_COMMON_SRCS flashattnv3.cc)
  list(APPEND DYNLOAD_COMMON_SRCS flashmaskv2.cc)
endif()

if(MKL_FOUND AND WITH_ONEMKL)
  message("ONEMKL INCLUDE directory is ${MKL_INCLUDE}")
  list(APPEND DYNLOAD_COMMON_SRCS mklrt.cc)
endif()

if(WITH_ROCM)
  collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS} ${HIP_SRCS})
elseif(WITH_GPU)
  if(WIN32)
    nv_library(
      dynload_common static
      SRCS ${DYNLOAD_COMMON_SRCS}
      DEPS warpctc)
    collect_srcs(backends_srcs SRCS ${CUDA_SRCS})
  else()
    collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS} ${CUDA_SRCS})
  endif()
elseif(WITH_XPU_FFT)
  collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS} ${XPU_SRCS})
else()
  if(WIN32)
    cc_library(
      dynload_common static
      SRCS ${DYNLOAD_COMMON_SRCS}
      DEPS warpctc)
  else()
    collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS})
  endif()
endif()

if(WITH_CUDNN_FRONTEND)
  nv_test(
    cudnn_frontend_test
    SRCS cudnn_frontend_test.cc
    DEPS phi common cudnn-frontend)
endif()
