

file(GLOB jit_kernel_mix_cc RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")
cc_library(jit_kernel_mix SRCS ${jit_kernel_mix_cc} DEPS jit_kernel_base)

set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} jit_kernel_mix PARENT_SCOPE)

USE_JITKERNEL_MORE(kVSigmoid, mix)
USE_JITKERNEL_MORE(kVTanh, mix)
USE_JITKERNEL_MORE(kLSTMCtHt, mix)
USE_JITKERNEL_MORE(kLSTMC1H1, mix)
USE_JITKERNEL_MORE(kGRUH1, mix)
USE_JITKERNEL_MORE(kGRUHtPart1, mix)
USE_JITKERNEL_MORE(kGRUHtPart2, mix)
USE_JITKERNEL_MORE(kSoftmax, mix)
