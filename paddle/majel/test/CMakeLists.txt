cc_test(place_test
    SRCS place_test.cc
    DEPS majel)

if(WITH_GPU)
    nv_test(cuda_test SRCS cuda_test.cu)
endif()
