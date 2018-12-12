
cc_library(jit_kernel_refer SRCS refer.cc DEPS jit_kernel_base)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} jit_kernel_refer PARENT_SCOPE)

function(USE_JITKERNEL_REFER TARGET)
    file(APPEND ${jit_file} "USE_JITKERNEL_REFER(${TARGET});\n")
endfunction()

# use refer kernel by name
USE_JITKERNEL_REFER(vmul)
USE_JITKERNEL_REFER(vadd)
USE_JITKERNEL_REFER(vaddrelu)
USE_JITKERNEL_REFER(vsub)
USE_JITKERNEL_REFER(vscal)
USE_JITKERNEL_REFER(vaddbias)
