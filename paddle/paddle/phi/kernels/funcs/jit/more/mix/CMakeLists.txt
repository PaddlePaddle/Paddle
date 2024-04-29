file(
  GLOB jit_kernel_mix_cc
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")

collect_srcs(kernels_srcs SRCS ${jit_kernel_mix_cc})

use_jitkernel_more(kVSigmoid, mix)
use_jitkernel_more(kVTanh, mix)
use_jitkernel_more(kLSTMCtHt, mix)
use_jitkernel_more(kLSTMC1H1, mix)
use_jitkernel_more(kGRUH1, mix)
use_jitkernel_more(kGRUHtPart1, mix)
use_jitkernel_more(kGRUHtPart2, mix)
