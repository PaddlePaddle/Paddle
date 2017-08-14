if(WITH_MKLML)
    set(BLAS_LIB mklml)
else()
    set(BLAS_LIB cblas)
endif()

if(WITH_GPU)
    nv_library(math_function SRCS math_function.cc math_function.cu DEPS ${BLAS_LIB} device_context)
else()
    cc_library(math_function SRCS math_function.cc DEPS ${BLAS_LIB} device_context)
endif()

nv_test(math_function_test SRCS math_function_test.cc DEPS math_function tensor)
