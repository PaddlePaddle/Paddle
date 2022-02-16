if (NOT INFRT_WITH_PTEN)
    return()
endif()

core_gather_headers()

gather_srcs(infrt_src SRCS
    registry.cc
    dense_tensor_kernels.cc
    context_kernels.cc
    allocator_kernels.cc
)

cc_library(infrt_naive SRCS infershaped/infershaped_kernel_launcher.cc
        infershaped/infershaped_kernel_launchers.cc
        )

cc_test_tiny(test_infrt_infershape_launchers SRCS
infershaped/infershape_launchers_test.cc DEPS infrt)
