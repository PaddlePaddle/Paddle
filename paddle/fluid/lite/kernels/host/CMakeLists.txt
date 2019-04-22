cc_library(fc_compute_host SRCS fc_compute.cc DEPS ${lite_kernel_deps})
cc_library(relu_compute_host SRCS relu_compute.cc DEPS ${lite_kernel_deps})
cc_library(mul_compute_host SRCS mul_compute.cc DEPS ${lite_kernel_deps})
cc_library(scale_compute_host SRCS scale_compute.cc DEPS ${lite_kernel_deps})
cc_library(feed_compute_host SRCS feed_compute.cc DEPS ${lite_kernel_deps})
cc_library(fetch_compute_host SRCS fetch_compute.cc DEPS ${lite_kernel_deps})

cc_library(host_kernels DEPS
        feed_compute_host
        fetch_compute_host
        fc_compute_host
        relu_compute_host
        mul_compute_host
        scale_compute_host
        DEPS ${lite_kernel_deps}
        )

cc_test(test_fc_compute SRCS fc_compute_test.cc DEPS fc_compute_host fc_op_lite)
