
cc_library(jit_kernel_mkl SRCS mkl.cc DEPS jit_kernel_base dynload_mklml)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} dynload_mklml jit_kernel_mkl PARENT_SCOPE)

# use mkl kernels by name and type
USE_JITKERNEL_MORE(vmul, mkl)
USE_JITKERNEL_MORE(vadd, mkl)
USE_JITKERNEL_MORE(vscal, mkl)
USE_JITKERNEL_MORE(vexp, mkl)
USE_JITKERNEL_MORE(vsigmoid, mkl)
USE_JITKERNEL_MORE(vtanh, mkl)
