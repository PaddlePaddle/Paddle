cc_library(place OBJECT
    SRCS place.cc)
cc_test(place_test
    SRCS place_test.cc
    OBJS place
    DEPS glog gflags)

cc_library(ddim OBJECT
    SRCS ddim.cc)
cc_test(ddim_test
    SRCS ddim_test.cc
    OBJS ddim)
nv_test(dim_test
    SRCS dim_test.cu
    OBJS ddim)

cc_library(majel
    OBJS place ddim)
