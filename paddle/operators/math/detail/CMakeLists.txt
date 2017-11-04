if(WITH_AVX)
    cc_library(activation_functions SRCS avx_functions.cc)
endif()
