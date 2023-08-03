add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(detail)
add_subdirectory(jit)

file(
  GLOB func_cc_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
# TODO(@caizhi): enable compiling all cu kernels
#if(WITH_GPU OR WITH_ROCM OR WITH_MUSA)
#  file(
#    GLOB func_cu_srcs
#    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
#    "*.cu")
#endif()
if(WITH_GPU OR WITH_ROCM OR WITH_MUSA)
  file(
    GLOB func_cu_srcs
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "concat_and_split_functor.cu"
    "math_function.cu"
    "segment_pooling.cu"
    "sequence_pooling.cu"
    "softmax.cu"
    "matrix_inverse.cu"
    "im2col.cu"
    "selected_rows_functor.cu"
    "gather_scatter_functor.cu")
endif()

collect_srcs(kernels_srcs SRCS ${func_cc_srcs} ${func_cu_srcs})
