if(NOT LITE_WITH_X86)
    return()
endif()

cc_library(activation_compute SRCS activation_compute.cc DEPS ${lite_kernel_deps} activation_op)
cc_library(elementwise_compute SRCS elementwise_compute.cc DEPS ${lite_kernel_deps})
cc_library(mean_compute SRCS mean_compute.cc DEPS ${lite_kernel_deps})
cc_library(fill_constant_compute SRCS fill_constant_compute.cc DEPS ${lite_kernel_deps})
cc_library(mul_compute SRCS mul_compute.cc DEPS ${lite_kernel_deps})
cc_library(sgd_compute SRCS sgd_compute.cc DEPS ${lite_kernel_deps})

set(x86_kernels
    activation_compute
    elementwise_compute
    mean_compute
    fill_constant_compute
    mul_compute
    )

set(x86_kernels "${x86_kernels}" CACHE INTERNAL "x86 kernels")
