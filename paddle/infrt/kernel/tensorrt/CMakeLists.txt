if (NOT (INFRT_WITH_PHI AND INFRT_WITH_GPU AND INFRT_WITH_TRT))
  return()
endif()

core_gather_headers()

gather_srcs(infrt_src SRCS
    registry.cc
    trt_kernels.cc
)
