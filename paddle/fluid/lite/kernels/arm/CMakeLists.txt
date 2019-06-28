if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
    return()
endif()

message(STATUS "compile with lite ARM kernels")

lite_cc_library(fc_compute_arm SRCS fc_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(activation_compute_arm SRCS activation_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(mul_compute_arm SRCS mul_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(scale_compute_arm SRCS scale_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(softmax_compute_arm SRCS softmax_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(conv_compute_arm SRCS conv_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(batch_norm_compute_arm SRCS batch_norm_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(elementwise_compute_arm SRCS elementwise_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(pool_compute_arm SRCS pool_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(split_compute_arm SRCS split_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(concat_compute_arm SRCS concat_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(dropout_compute_arm SRCS dropout_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(calib_compute_arm SRCS calib_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(transpose_compute_arm SRCS transpose_compute.cc DEPS ${lite_kernel_deps} math_arm)

lite_cc_test(test_fc_compute_arm SRCS fc_compute_test.cc DEPS fc_compute_arm math_arm)
lite_cc_test(test_activation_compute_arm SRCS activation_compute_test.cc DEPS activation_compute_arm)
lite_cc_test(test_scale_compute_arm SRCS scale_compute_test.cc DEPS scale_compute_arm)
lite_cc_test(test_softmax_compute_arm SRCS softmax_compute_test.cc DEPS softmax_compute_arm)
lite_cc_test(test_conv_compute_arm SRCS conv_compute_test.cc DEPS conv_compute_arm)
lite_cc_test(test_batch_norm_compute_arm SRCS batch_norm_compute_test.cc DEPS batch_norm_compute_arm)
lite_cc_test(test_elementwise_compute_arm SRCS elementwise_compute_test.cc DEPS elementwise_compute_arm)
lite_cc_test(test_pool_compute_arm SRCS pool_compute_test.cc DEPS pool_compute_arm)
lite_cc_test(test_mul_compute_arm SRCS mul_compute_test.cc DEPS mul_compute_arm)
lite_cc_test(test_split_compute_arm SRCS split_compute_test.cc DEPS split_compute_arm)
lite_cc_test(test_concat_compute_arm SRCS concat_compute_test.cc DEPS concat_compute_arm)
lite_cc_test(test_dropout_compute_arm SRCS dropout_compute_test.cc DEPS dropout_compute_arm)
# lite_cc_test(test_calib_compute_arm SRCS calib_compute_test.cc DEPS calib_compute_arm)
lite_cc_test(test_transpose_compute_arm SRCS transpose_compute_test.cc DEPS transpose_compute_arm)

set(arm_kernels
    fc_compute_arm
    activation_compute_arm
    mul_compute_arm
    scale_compute_arm
    softmax_compute_arm
    conv_compute_arm
    batch_norm_compute_arm
    elementwise_compute_arm
    pool_compute_arm
    split_compute_arm
    concat_compute_arm
    dropout_compute_arm
    transpose_compute_arm
    calib_compute_arm
    )

set(arm_kernels "${arm_kernels}" CACHE INTERNAL "arm kernels")
