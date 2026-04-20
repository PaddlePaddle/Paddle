add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)
if(WITH_MAGMA)
  add_subdirectory(magma)
endif()
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

# Remove fast_ln_v2 kernels on HIP and Windows platforms
if(WITH_ROCM OR WIN32)
  list(REMOVE_ITEM func_cu_srcs "fast_ln_v2_bwd_kernel.cu")
  list(REMOVE_ITEM func_cu_srcs "fast_ln_v2_fwd_kernel.cu")
endif()

collect_srcs(kernels_srcs SRCS ${func_cc_srcs} ${func_cu_srcs})
