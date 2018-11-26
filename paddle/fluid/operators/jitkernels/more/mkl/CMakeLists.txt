
cc_library(jit_kernel_mkl SRCS mkl.cc DEPS jit_kernel_base dynload_mklml)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} dynload_mklml jit_kernel_mkl PARENT_SCOPE)
