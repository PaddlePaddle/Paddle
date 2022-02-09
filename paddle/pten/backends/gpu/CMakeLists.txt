if(WITH_GPU)
  add_subdirectory(cuda)
  nv_library(pten_gpu_info SRCS gpu_info.cc DEPS pten_cuda_info gflags glog enforce pten_dynload_cuda)
elseif(WITH_ROCM)
  add_subdirectory(rocm)
  hip_library(pten_gpu_info SRCS gpu_info.cc DEPS pten_rocm_info gflags glog enforce pten_dynload_cuda)
endif()

cc_library(gpu_context SRCS gpu_context.cc DEPS pten_device_context pten_gpu_info eigen3)
