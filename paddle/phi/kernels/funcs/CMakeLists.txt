add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(detail)
add_subdirectory(jit)

file(
  GLOB func_cc_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
if(WITH_GPU OR WITH_ROCM OR WITH_MUSA)
  file(
    GLOB func_cu_srcs
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cu")
endif()

collect_srcs(kernels_srcs SRCS ${func_cc_srcs} ${func_cu_srcs})
