
if(WITH_GPU)
    nv_library(math_function SRCS math_function.cc math_function.cu DEPS cblas device_context eigen3)
else()
    cc_library(math_function SRCS math_function.cc DEPS cblas device_context eigen3)
endif()

nv_test(math_function_test SRCS math_function_test.cc DEPS math_function tensor)
