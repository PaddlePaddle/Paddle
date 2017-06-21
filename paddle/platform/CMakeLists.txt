nv_test(cuda_test SRCS cuda_test.cu)

cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)
