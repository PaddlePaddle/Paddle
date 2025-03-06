set(SRCS deep_ep.cpp kernels/internode.cu kernels/intranode.cu
         kernels/runtime.cu)

cc_library(
  deep_ep
  SRCS ${SRCS}
  DEPS phi common)

# Link CPP and CUDA together
target_compile_options(deep_ep PRIVATE -Wno-reorder)
