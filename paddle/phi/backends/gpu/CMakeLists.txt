if(WITH_GPU)
  add_subdirectory(cuda)
  nv_library(phi_gpu_info SRCS gpu_info.cc DEPS phi_cuda_info gflags glog enforce phi_dynload_cuda)
elseif(WITH_ROCM)
  add_subdirectory(rocm)
  hip_library(phi_gpu_info SRCS gpu_info.cc DEPS phi_rocm_info gflags glog enforce phi_dynload_cuda)
endif()

cc_library(gpu_context SRCS gpu_context.cc DEPS phi_device_context phi_gpu_info eigen3)
