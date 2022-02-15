if (NOT INFRT_WITH_PTEN)
    return()
endif()

core_gather_headers()

gather_srcs(infrt_src SRCS
    registry.cc
    dense_tensor_kernels.cc
)
