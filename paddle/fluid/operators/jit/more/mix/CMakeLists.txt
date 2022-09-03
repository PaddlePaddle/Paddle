file(
  GLOB jit_kernel_mix_cc
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
cc_library(
  jit_kernel_mix
  SRCS ${jit_kernel_mix_cc}
  DEPS jit_kernel_base)

set(JIT_KERNEL_DEPS
    ${JIT_KERNEL_DEPS} jit_kernel_mix
    PARENT_SCOPE)

use_jitkernel_more(kVSigmoid, mix)
use_jitkernel_more(kVTanh, mix)
use_jitkernel_more(kLSTMCtHt, mix)
use_jitkernel_more(kLSTMC1H1, mix)
use_jitkernel_more(kGRUH1, mix)
use_jitkernel_more(kGRUHtPart1, mix)
use_jitkernel_more(kGRUHtPart2, mix)
use_jitkernel_more(kSoftmax, mix)
