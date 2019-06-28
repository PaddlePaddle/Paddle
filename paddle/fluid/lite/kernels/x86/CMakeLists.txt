if(NOT LITE_WITH_X86)
    return()
endif()

lite_cc_library(activation_compute_x86 SRCS activation_compute.cc DEPS ${lite_kernel_deps} activation_op)
lite_cc_library(mean_compute_x86 SRCS mean_compute.cc DEPS ${lite_kernel_deps})
lite_cc_library(fill_constant_compute_x86 SRCS fill_constant_compute.cc DEPS ${lite_kernel_deps})
lite_cc_library(sgd_compute_x86 SRCS sgd_compute.cc DEPS ${lite_kernel_deps})

lite_cc_library(fc_compute_x86 SRCS fc_compute.cc DEPS ${lite_kernel_deps})
lite_cc_library(mul_compute_x86 SRCS mul_compute.cc DEPS ${lite_kernel_deps})
lite_cc_library(relu_compute_x86 SRCS relu_compute.cc DEPS ${lite_kernel_deps})
lite_cc_library(scale_compute_x86 SRCS scale_compute.cc DEPS ${lite_kernel_deps})
lite_cc_library(elementwise_compute_x86 SRCS elementwise_compute.cc DEPS ${lite_kernel_deps} elementwise_sub_op elementwise_add_op)
lite_cc_library(softmax_compute_x86 SRCS softmax_compute.cc DEPS ${lite_kernel_deps} softmax)
lite_cc_library(dropout_compute_x86 SRCS dropout_compute.cc DEPS ${lite_kernel_deps} )
lite_cc_library(concat_compute_x86 SRCS concat_compute.cc DEPS ${lite_kernel_deps} )
lite_cc_library(conv_compute_x86 SRCS conv_compute.cc DEPS ${lite_kernel_deps} blas im2col vol2col)
lite_cc_library(pool_compute_x86 SRCS pool_compute.cc DEPS ${lite_kernel_deps} pooling)
lite_cc_library(batch_norm_compute_x86 SRCS batch_norm_compute.cc DEPS ${lite_kernel_deps})
lite_cc_library(uniform_random_compute_x86 SRCS uniform_random_compute.cc DEPS ${lite_kernel_deps} )

lite_cc_test(test_fc_compute_x86 SRCS fc_compute_test.cc DEPS fc_compute_x86)
lite_cc_test(test_conv2d_compute_x86 SRCS conv_compute_test.cc DEPS conv_compute_x86)
lite_cc_test(test_pool2d_compute_x86 SRCS pool_compute_test.cc DEPS pool_compute_x86)
lite_cc_test(test_concat_compute_x86 SRCS concat_compute_test.cc DEPS concat_compute_x86)
lite_cc_test(test_softmax_compute_x86 SRCS softmax_compute_test.cc DEPS softmax_compute_x86)
lite_cc_test(test_elementwise_compute_x86 SRCS elementwise_compute_test.cc DEPS elementwise_compute_x86)
lite_cc_test(test_relu_compute_x86 SRCS relu_compute_test.cc DEPS relu_compute_x86)
lite_cc_test(test_mul_compute_x86 SRCS mul_compute_test.cc DEPS mul_compute_x86 operator)
lite_cc_test(test_scale_compute_x86 SRCS scale_compute_test.cc DEPS scale_compute_x86)
lite_cc_test(test_dropout_compute_x86 SRCS dropout_compute_test.cc DEPS dropout_compute_x86)
lite_cc_test(test_batch_norm_compute_x86 SRCS batch_norm_compute_test.cc DEPS batch_norm_compute_x86)


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
    batch_norm_compute_x86
    uniform_random_compute_x86
    sgd_compute_x86
    CACHE INTERNAL "x86 kernels")
    
