
cc_library(jit_kernel_jitcode SRCS jitcode.cc DEPS jit_kernel_base xbyak)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} xbyak jit_kernel_jitcode PARENT_SCOPE)
