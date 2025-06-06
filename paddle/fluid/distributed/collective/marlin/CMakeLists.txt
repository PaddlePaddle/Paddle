set(SRCS ops.cu)
add_subdirectory(kernels)

cc_library(
  marlin
  SRCS ${SRCS}
  DEPS phi common kernels_lib)


# Link CPP and CUDA together
target_compile_options(marlin PRIVATE -Wno-reorder)
