
cc_library(jit_kernel_refer SRCS refer.cc DEPS jit_kernel_base)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} jit_kernel_refer PARENT_SCOPE)
