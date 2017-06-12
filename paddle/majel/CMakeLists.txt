cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)

if (WITH_GPU)
  nv_library(malloc SRCS malloc.cc DEPS glog gflags)
else()
  cc_library(malloc SRCS malloc.cc DEPS glog gflags)
endif()
cc_library(allocation SRCS allocation.cc)
cc_test(allocation_test SRCS allocation_test.cc DEPS allocation malloc place glog gflags)

nv_test(cuda_test SRCS cuda_test.cu)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)
