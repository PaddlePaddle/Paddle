add_subdirectory(dynload)

nv_test(cuda_test SRCS cuda_test.cu)

cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

nv_test(device_context_test SRCS device_context_test.cc DEPS dynload_cuda dynamic_loader eigen3 place)
