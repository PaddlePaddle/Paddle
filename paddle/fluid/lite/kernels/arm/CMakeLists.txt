if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

message(STATUS "compile with lite ARM kernels")

cc_library(fc_compute_arm SRCS fc_compute.cc DEPS ${lite_kernel_deps} eigen3)
cc_library(relu_compute_arm SRCS relu_compute.cc DEPS ${lite_kernel_deps})
cc_library(mul_compute_arm SRCS mul_compute.cc DEPS ${lite_kernel_deps} eigen3)
cc_library(scale_compute_arm SRCS scale_compute.cc DEPS ${lite_kernel_deps} eigen3)

cc_library(feed_compute_arm SRCS feed_compute.cc DEPS ${lite_kernel_deps})
cc_library(fetch_compute_arm SRCS fetch_compute.cc DEPS ${lite_kernel_deps})

# lite_cc_test(test_fc_compute_arm SRCS fc_compute_test.cc DEPS ${lite_kernel_deps} fc_compute_arm)

set(arm_kernels
    feed_compute_arm
    fetch_compute_arm
    fc_compute_arm
    relu_compute_arm
    mul_compute_arm
    scale_compute_arm
    )

set(arm_kernels "${arm_kernels}" CACHE INTERNAL "arm kernels")
 
