message(STATUS "compile with lite host kernels")

cc_library(feed_compute_host SRCS feed_compute.cc DEPS ${lite_kernel_deps})
cc_library(fetch_compute_host SRCS fetch_compute.cc DEPS ${lite_kernel_deps})
cc_library(reshape_compute_host SRCS reshape_compute.cc DEPS ${lite_kernel_deps} reshape_op_lite)

lite_cc_test(test_reshape_compute_host SRCS reshape_compute_test.cc DEPS reshape_compute_host)

set(host_kernels
    feed_compute_host
    fetch_compute_host
    reshape_compute_host
    )

set(host_kernels "${host_kernels}" CACHE INTERNAL "host kernels")
 
