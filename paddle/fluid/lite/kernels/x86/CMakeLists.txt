if(NOT LITE_WITH_X86)
    return()
endif()

cc_library(activation_compute_x86 SRCS activation_compute.cc DEPS ${lite_kernel_deps} activation_op)
cc_library(mean_compute_x86 SRCS mean_compute.cc DEPS ${lite_kernel_deps})
cc_library(fill_constant_compute_x86 SRCS fill_constant_compute.cc DEPS ${lite_kernel_deps})
cc_library(sgd_compute_x86 SRCS sgd_compute.cc DEPS ${lite_kernel_deps})

cc_library(fc_compute_x86 SRCS fc_compute.cc DEPS ${lite_kernel_deps})
cc_library(mul_compute_x86 SRCS mul_compute.cc DEPS ${lite_kernel_deps})
cc_library(relu_compute_x86 SRCS relu_compute.cc DEPS ${lite_kernel_deps})
cc_library(scale_compute_x86 SRCS scale_compute.cc DEPS ${lite_kernel_deps})
cc_library(elementwise_compute_x86 SRCS elementwise_compute.cc DEPS ${lite_kernel_deps} elementwise_sub_op elementwise_add_op)
cc_library(softmax_compute_x86 SRCS softmax_compute.cc DEPS ${lite_kernel_deps} softmax)
cc_library(dropout_compute_x86 SRCS dropout_compute.cc DEPS ${lite_kernel_deps} )
cc_library(concat_compute_x86 SRCS concat_compute.cc DEPS ${lite_kernel_deps} )
cc_library(conv_compute_x86 SRCS conv_compute.cc DEPS ${lite_kernel_deps} blas im2col vol2col)
cc_library(pool_compute_x86 SRCS pool_compute.cc DEPS ${lite_kernel_deps} pooling)

set(x86_kernels
    activation_compute_x86
    elementwise_compute_x86
    mean_compute_x86
    fill_constant_compute_x86
    mul_compute_x86
    relu_compute_x86
    fc_compute_x86
    scale_compute_x86
    softmax_compute_x86
    dropout_compute_x86
    concat_compute_x86
    conv_compute_x86 
    pool_compute_x86  
    )

set(x86_kernels "${x86_kernels}" CACHE INTERNAL "x86 kernels")
 
