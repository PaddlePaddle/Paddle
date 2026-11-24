if(WITH_GPU)
  nv_test(
    gin_overlap_sum_test
    SRCS gin_overlap_sum_test.cu
    DEPS gin_overlap_kernels)
endif()
