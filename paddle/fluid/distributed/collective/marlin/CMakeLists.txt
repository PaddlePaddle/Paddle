set(SRCS ops.cu)

cc_library(
  marlin
  SRCS ${SRCS}
  DEPS phi common)

# Link CPP and CUDA together
target_compile_options(marlin PRIVATE -Wno-reorder)