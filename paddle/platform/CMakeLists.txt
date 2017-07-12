add_subdirectory(dynload)

nv_test(cuda_test SRCS cuda_test.cu)

cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

IF(WITH_GPU)
    set(GPU_CTX_DEPS dynload_cuda dynamic_loader)
ELSE()
    set(GPU_CTX_DEPS)
ENDIF()

cc_library(device_context SRCS device_context.cc DEPS place eigen3 ${GPU_CTX_DEPS})
nv_test(device_context_test SRCS device_context_test.cc DEPS device_context glog gflags)
