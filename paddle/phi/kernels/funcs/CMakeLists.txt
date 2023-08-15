add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(detail)
add_subdirectory(jit)

file(
  GLOB func_cc_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
if(WITH_GPU
   OR WITH_ROCM
   OR WITH_MUSA)
  file(
    GLOB func_cu_srcs
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cu")
endif()

# TODO(@MTAI): compilation error will occur when compiling the following files.
# Compiler mcc need fix this bug.
if(WITH_MUSA)
  list(REMOVE_ITEM func_cu_srcs "cross_entropy.cu" "gru_compute.cu"
       "softmax.cu")
endif()

collect_srcs(kernels_srcs SRCS ${func_cc_srcs} ${func_cu_srcs})
