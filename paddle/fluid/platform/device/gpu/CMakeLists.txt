if(WITH_GPU)
  add_subdirectory(cuda)
  nv_library(
    gpu_info
    SRCS gpu_info.cc
    DEPS phi common glog enforce monitor)

  nv_test(cuda_helper_test SRCS cuda_helper_test.cu)
  nv_test(
    cudnn_desc_test
    SRCS cudnn_desc_test.cc
    DEPS phi common)
elseif(WITH_ROCM)
  add_subdirectory(rocm)
  hip_library(
    gpu_info
    SRCS gpu_info.cc
    DEPS phi common glog enforce monitor)

  hip_test(cuda_helper_test SRCS cuda_helper_test.cu)
  hip_test(
    cudnn_desc_test
    SRCS cudnn_desc_test.cc
    DEPS phi common)
endif()

cc_library(
  gpu_resource_pool
  SRCS gpu_resource_pool.cc
  DEPS gpu_info)
