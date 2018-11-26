
set(JIT_KERNEL_DEPS cpu_info cblas gflags enforce place)

cc_library(jit_kernel_base SRCS kernels.cc DEPS ${JIT_KERNEL_DEPS})

add_subdirectory(more)
add_subdirectory(refer)

if(WITH_XBYAK)
    add_subdirectory(jitcode)
endif()

# Debug
message(STATUS "--------${JIT_KERNEL_DEPS}")

cc_library(jit_kernel SRCS kernels.cc DEPS ${JIT_KERNEL_DEPS})
cc_test(jit_kernel_test SRCS test.cc DEPS jit_kernel)
