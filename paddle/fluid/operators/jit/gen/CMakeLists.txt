
file(GLOB jitcode_cc_srcs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

cc_library(jit_kernel_jitcode SRCS ${jitcode_cc_srcs} DEPS jit_kernel_base xbyak)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} xbyak jit_kernel_jitcode PARENT_SCOPE)
