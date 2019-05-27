if(NOT LITE_WITH_X86)
    return()
endif()

cc_library(activation_compute_x86 SRCS activation_compute.cc DEPS ${lite_kernel_deps} activation_op)
cc_library(elementwise_compute_x86 SRCS elementwise_compute.cc DEPS ${lite_kernel_deps})
cc_library(mean_compute_x86 SRCS mean_compute.cc DEPS ${lite_kernel_deps})
cc_library(fill_constant_compute_x86 SRCS fill_constant_compute.cc DEPS ${lite_kernel_deps})
cc_library(mul_compute_x86 SRCS mul_compute.cc DEPS ${lite_kernel_deps})
cc_library(sgd_compute_x86 SRCS sgd_compute.cc DEPS ${lite_kernel_deps})

set(x86_kernels
    activation_compute_x86
    elementwise_compute_x86
    mean_compute_x86
    fill_constant_compute_x86
    mul_compute_x86
    )

set(x86_kernels "${x86_kernels}" CACHE INTERNAL "x86 kernels")
