if(NOT LITE_WITH_X86)
    return()
endif()

cc_library(activation_compute SRCS activation_compute.cc DEPS ${lite_kernel_deps} activation_op)
cc_library(elementwise_compute SRCS elementwise_compute.cc DEPS ${lite_kernel_deps} elementwise_sub_op)
