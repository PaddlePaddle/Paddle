set(COLLECTIVE_COMMON_SRCS common.cc process_group.cc process_group_nccl.cc)

collect_srcs(core_srcs SRCS ${COLLECTIVE_COMMON_SRCS})
