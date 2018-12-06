

if(WITH_MKLML)
    add_subdirectory(mkl)
endif()

set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} PARENT_SCOPE)
