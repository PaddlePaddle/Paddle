cc_library(
  jit_kernel_mkl
  SRCS mkl.cc
  DEPS jit_kernel_base dynload_mklml)
set(JIT_KERNEL_DEPS
    ${JIT_KERNEL_DEPS} dynload_mklml jit_kernel_mkl
    PARENT_SCOPE)

# use mkl kernels by name and type
use_jitkernel_more(kMatMul, mkl)
use_jitkernel_more(kVMul, mkl)
use_jitkernel_more(kVAdd, mkl)
use_jitkernel_more(kVScal, mkl)
use_jitkernel_more(kStrideScal, mkl)
use_jitkernel_more(kVExp, mkl)
use_jitkernel_more(kVSquare, mkl)
use_jitkernel_more(kVCopy, mkl)
use_jitkernel_more(kVSigmoid, mkl)
use_jitkernel_more(kVTanh, mkl)
use_jitkernel_more(kSeqPool, mkl)
use_jitkernel_more(kSoftmax, mkl)
use_jitkernel_more(kEmbSeqPool, mkl)
use_jitkernel_more(kSgd, mkl)
use_jitkernel_more(kVBroadcast, mkl)
