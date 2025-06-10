set(MARLIN_KERNEL_SRCS kernels/kernel_fp16_ku4.cu gptq_marlin/kernel_fp16_ku4.cu)
nv_library(marlin_kernels SRCS ${MARLIN_KERNEL_SRCS})

nv_library(
  marlin
  SRCS ops.cu
  DEPS phi common marlin_kernels)

set_target_properties(marlin PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(marlin PRIVATE -Wno-reorder -Wno-unused-variable)