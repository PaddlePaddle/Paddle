if(WITH_GPU)
    nv_library(math_function SRCS math_function.cc math_function.cu im2col.cc
      im2col.cu DEPS cblas device_context operator)
    nv_library(softmax SRCS softmax.cc softmax.cu DEPS operator)
    nv_library(cross_entropy SRCS cross_entropy.cc cross_entropy.cu
      DEPS operator)
else()
    cc_library(math_function SRCS math_function.cc im2col.cc
      DEPS cblas device_context operator)
    cc_library(softmax SRCS softmax.cc DEPS operator)
    cc_library(cross_entropy SRCS cross_entropy.cc DEPS operator)
endif()

nv_test(math_function_test SRCS math_function_test.cc DEPS math_function tensor)
cc_test(im2col_test SRCS im2col_test.cc DEPS math_function tensor)
