add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(detail)
add_subdirectory(jit)
add_subdirectory(math)

file(
  GLOB func_cc_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
if(WITH_GPU OR WITH_ROCM)
  file(
    GLOB func_cu_srcs
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cu")
endif()

# Note(qili93): remove kernels not supported on DCU yet
if(WITH_ROCM)
  list(REMOVE_ITEM func_cu_srcs "weight_only_gemv.cu")
endif()

collect_srcs(kernels_srcs SRCS ${func_cc_srcs} ${func_cu_srcs})
