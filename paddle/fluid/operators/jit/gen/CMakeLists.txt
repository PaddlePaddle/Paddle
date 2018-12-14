
file(GLOB jitcode_cc_srcs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

cc_library(jit_kernel_jitcode SRCS ${jitcode_cc_srcs} DEPS jit_kernel_base xbyak)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} xbyak jit_kernel_jitcode PARENT_SCOPE)

function(USE_JITKERNEL_GEN TARGET)
    file(APPEND ${jit_file} "USE_JITKERNEL_GEN(${TARGET});\n")
endfunction()

# use gen jitcode kernel by name
USE_JITKERNEL_GEN(vmul)
USE_JITKERNEL_GEN(vadd)
#USE_JITKERNEL_GEN(vsub) # TODO(TJ): enable me
USE_JITKERNEL_GEN(vaddrelu)
USE_JITKERNEL_GEN(vscal)
USE_JITKERNEL_GEN(vaddbias)
USE_JITKERNEL_GEN(vrelu)
USE_JITKERNEL_GEN(videntity)
USE_JITKERNEL_GEN(vexp)
USE_JITKERNEL_GEN(vsigmoid)
USE_JITKERNEL_GEN(vtanh)
USE_JITKERNEL_GEN(lstmctht)
USE_JITKERNEL_GEN(lstmc1h1)
