if(WITH_GPU)
    nv_library(math_function SRCS math_function.cc math_function.cu im2col.cc
      im2col.cu softmax_function.cc DEPS cblas device_context operator)
else()
    cc_library(math_function SRCS math_function.cc im2col.cc
      softmax_function.cc DEPS cblas device_context operator)
endif()

nv_test(math_function_test SRCS math_function_test.cc DEPS math_function tensor)
cc_test(im2col_test SRCS im2col_test.cc DEPS math_function tensor)
