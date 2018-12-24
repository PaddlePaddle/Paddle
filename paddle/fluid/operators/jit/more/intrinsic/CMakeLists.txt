
file(GLOB jit_kernel_cc_intrinsic RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")
cc_library(jit_kernel_intrinsic SRCS ${jit_kernel_cc_intrinsic} DEPS jit_kernel_base)

set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} jit_kernel_intrinsic PARENT_SCOPE)

# use mkl kernels by name and type
USE_JITKERNEL_MORE(kCRFDecoding, intrinsic)
USE_JITKERNEL_MORE(kLayerNorm, intrinsic)
