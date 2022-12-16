file(
  GLOB jitcode_cc_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")

cc_library(
  jit_kernel_jitcode
  SRCS ${jitcode_cc_srcs}
  DEPS jit_kernel_base xbyak)
set(JIT_KERNEL_DEPS
    ${JIT_KERNEL_DEPS} xbyak jit_kernel_jitcode
    PARENT_SCOPE)

function(USE_JITKERNEL_GEN TARGET)
  file(APPEND ${jit_file} "USE_JITKERNEL_GEN(${TARGET});\n")
endfunction()

# use gen jitcode kernel by name
use_jitkernel_gen(kMatMul)
use_jitkernel_gen(kVMul)
use_jitkernel_gen(kVAdd)
use_jitkernel_gen(kVSub)
use_jitkernel_gen(kVAddRelu)
use_jitkernel_gen(kVScal)
use_jitkernel_gen(kVAddBias)
use_jitkernel_gen(kVRelu)
use_jitkernel_gen(kVSquare)
use_jitkernel_gen(kVIdentity)
use_jitkernel_gen(kVExp)
use_jitkernel_gen(kVSigmoid)
use_jitkernel_gen(kVTanh)
use_jitkernel_gen(kLSTMCtHt)
use_jitkernel_gen(kLSTMC1H1)
use_jitkernel_gen(kGRUH1)
use_jitkernel_gen(kGRUHtPart1)
use_jitkernel_gen(kGRUHtPart2)
use_jitkernel_gen(kNCHW16CMulNC)
use_jitkernel_gen(kSeqPool)
use_jitkernel_gen(kHMax)
use_jitkernel_gen(kHSum)
use_jitkernel_gen(kEmbSeqPool)
use_jitkernel_gen(kAdam)
use_jitkernel_gen(kAdamW)
use_jitkernel_gen(kSgd)
use_jitkernel_gen(kVBroadcast)
