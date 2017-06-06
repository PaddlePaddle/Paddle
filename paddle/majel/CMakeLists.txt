cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)

nv_test(cuda_test SRCS cuda_test.cu)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)
