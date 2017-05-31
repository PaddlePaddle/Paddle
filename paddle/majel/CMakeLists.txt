cc_library(place OBJECT
    SRCS place.cc)
cc_library(ddim OBJECT
    SRCS ddim.cc)

cc_library(majel STATIC
    OBJS place ddim)

cc_test(place_test
    SRCS place_test.cc
    OBJS place)

nv_test(cuda_test
    SRCS cuda_test.cu)

nv_test(dim_test
    SRCS dim_test.cu
    OBJS ddim)
