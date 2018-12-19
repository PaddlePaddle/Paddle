

file(GLOB jit_kernel_mix_cc RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")
cc_library(jit_kernel_mix SRCS ${jit_kernel_mix_cc} DEPS jit_kernel_base)

set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} jit_kernel_mix PARENT_SCOPE)

USE_JITKERNEL_MORE(vsigmoid, mix)
USE_JITKERNEL_MORE(vtanh, mix)
USE_JITKERNEL_MORE(lstmctht, mix)
USE_JITKERNEL_MORE(lstmc1h1, mix)
USE_JITKERNEL_MORE(gruh1, mix)
USE_JITKERNEL_MORE(gruhtpart1, mix)
USE_JITKERNEL_MORE(gruhtpart2, mix)
