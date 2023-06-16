set(CHECK_COMMON_SRCS static_check.cc)

if(WITH_NCCL OR WITH_RCCL)
  list(APPEND CHECK_COMMON_SRCS nccl_dynamic_check.cc)
endif()

collect_srcs(core_srcs SRCS ${CHECK_COMMON_SRCS})
