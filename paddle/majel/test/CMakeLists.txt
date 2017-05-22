cc_test(place_test
    SRCS place_test.cc
    DEPS place)

cc_test(ddim_test
    SRCS ddim_test.cc
    DEPS ddim)

cc_test(allocation_test
    SRCS allocation_test.cc
    DEPS place malloc allocation)

if(WITH_GPU)
    nv_test(cuda_test SRCS cuda_test.cu)
    nv_test(dim_test SRCS dim_test.cu DEPS ddim)
endif()
