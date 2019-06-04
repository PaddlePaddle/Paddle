if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
    return()
endif()

message(STATUS "compile with lite ARM kernels")

cc_library(fc_compute_arm SRCS fc_compute.cc DEPS ${lite_kernel_deps} math_arm)
cc_library(relu_compute_arm SRCS relu_compute.cc DEPS ${lite_kernel_deps})
cc_library(mul_compute_arm SRCS mul_compute.cc DEPS ${lite_kernel_deps} eigen3)
cc_library(scale_compute_arm SRCS scale_compute.cc DEPS ${lite_kernel_deps} math_arm)
cc_library(softmax_compute_arm SRCS softmax_compute.cc DEPS ${lite_kernel_deps} math_arm)
cc_library(elementwise_add_compute_arm SRCS elementwise_add_compute.cc DEPS ${lite_kernel_deps})

lite_cc_test(test_fc_compute_arm SRCS fc_compute_test.cc DEPS fc_compute_arm math_arm)
lite_cc_test(test_scale_compute_arm SRCS scale_compute_test.cc DEPS scale_compute_arm)
lite_cc_test(test_softmax_compute_arm SRCS softmax_compute_test.cc DEPS softmax_compute_arm)
lite_cc_test(test_elementwise_add_compute_arm SRCS elementwise_add_compute_test.cc DEPS elementwise_add_compute_arm)

set(arm_kernels
    fc_compute_arm
    relu_compute_arm
    mul_compute_arm
    scale_compute_arm
    softmax_compute_arm
	elementwise_add_compute_arm)

set(arm_kernels "${arm_kernels}" CACHE INTERNAL "arm kernels")
