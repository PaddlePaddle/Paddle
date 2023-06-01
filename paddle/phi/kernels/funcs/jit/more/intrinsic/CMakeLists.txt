file(
  GLOB jit_kernel_cc_intrinsic
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")

collect_srcs(kernels_srcs SRCS ${jit_kernel_cc_intrinsic})

# use mkl kernels by name and type
use_jitkernel_more(kCRFDecoding, intrinsic)
use_jitkernel_more(kLayerNorm, intrinsic)
