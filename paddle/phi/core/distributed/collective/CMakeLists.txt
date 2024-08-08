set(COLLECTIVE_COMMON_SRCS common.cc process_group.cc)

if(WITH_NCCL OR WITH_RCCL)
  list(APPEND COLLECTIVE_COMMON_SRCS process_group_nccl.cc)
endif()

collect_srcs(core_srcs SRCS ${COLLECTIVE_COMMON_SRCS})
