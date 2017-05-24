cc_library(place SRCS place/place.cc)
cc_library(ddim SRCS ddim/ddim.cc)

if(WITH_TESTING)
    cc_test(ddim_test SRCS ddim/ddim_test.cc DEPS ddim)
    cc_test(place_test SRCS place/place_test.cc DEPS place)

    if(WITH_GPU)
        nv_test(cuda_test SRCS cuda_test/cuda_test.cu)
        nv_test(dim_test SRCS ddim/dim_test.cu DPES ddim)
    endif()

endif()
