if(WITH_GPU)
  cc_library(enforce SRCS enforce.cc DEPS nccl)
else()
  cc_library(enforce SRCS enforce.cc)
endif()
cc_test(enforce_test SRCS enforce_test.cc DEPS stringpiece enforce)

cc_library(cpu_info SRCS cpu_info.cc DEPS gflags glog enforce)
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info)

nv_library(gpu_info SRCS gpu_info.cc DEPS gflags glog enforce)

cc_library(place SRCS place.cc DEPS enforce)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

add_subdirectory(dynload)

IF(WITH_GPU)
    set(GPU_CTX_DEPS dynload_cuda dynamic_loader)
ELSE()
    set(GPU_CTX_DEPS)
ENDIF()

# memcpy deoends on device_context, here add deps individually for
# avoiding cycle dependencies
cc_library(device_context SRCS device_context.cc DEPS memory buddy_allocator
    system_allocator memory_block meta_data meta_cache place eigen3 ${GPU_CTX_DEPS})
nv_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info)

nv_test(cudnn_helper_test SRCS cudnn_helper_test.cc DEPS dynload_cuda)
nv_test(transform_test SRCS transform_test.cu DEPS paddle_memory place device_context)
nv_test(nccl_test SRCS nccl_test.cu DEPS dynload_cuda gpu_info device_context)
