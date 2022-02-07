core_gather_headers()

if (INFRT_WITH_PTEN)
  set(pten_kernel_src pten_kernels.cc)
endif()

gather_srcs(infrt_src SRCS
    basic_kernels.cc
    ${pten_kernel_src}
    test_kernels.cc
    tensor_shape_kernels.cc
    tensor_kernels.cc
    control_flow_kernels.cc
    )

if(INFRT_WITH_PTEN)
  add_subdirectory(pten)
endif()
