if(WITH_NCCL OR WITH_RCCL)
  set(SYMMETRIC_MEMORY_SRCS symmetric_memory.cc)
  set(SYMMETRIC_MEMORY_CU_SRCS symmetric_memory_kernels.cu)

  collect_srcs(core_srcs SRCS ${SYMMETRIC_MEMORY_SRCS})

  if(WITH_GPU)
    collect_srcs(core_srcs SRCS ${SYMMETRIC_MEMORY_CU_SRCS})
  endif()
endif()
