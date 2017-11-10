cc_library(cpu_info SRCS cpu_info.cc DEPS gflags glog)
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info)

nv_library(gpu_info SRCS gpu_info.cc DEPS gflags glog)

cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

add_subdirectory(dynload)

cc_test(enforce_test SRCS enforce_test.cc DEPS stringpiece)

IF(WITH_GPU)
    set(GPU_CTX_DEPS dynload_cuda dynamic_loader)
ELSE()
    set(GPU_CTX_DEPS)
ENDIF()

# memcpy deoends on device_context, here add deps individually for
# avoiding cycle dependencies
cc_library(device_context SRCS device_context.cc DEPS memory buddy_allocator
    system_allocator memory_block meta_data meta_cache place eigen3 ${GPU_CTX_DEPS})
nv_test(device_context_test SRCS device_context_test.cc DEPS device_context gpu_info)

nv_test(cudnn_helper_test SRCS cudnn_helper_test.cc DEPS dynload_cuda)
nv_test(transform_test SRCS transform_test.cu DEPS paddle_memory place device_context)
nv_test(nccl_test SRCS nccl_test.cu DEPS dynload_cuda gpu_info device_context)
