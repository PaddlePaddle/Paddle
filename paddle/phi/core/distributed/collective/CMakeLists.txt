set(COLLECTIVE_SRCS process_group.cc common.cc)

if(WITH_DISTRIBUTE)
  list(APPEND COLLECTIVE_SRCS process_group_gloo.cc gloo_send_recv.cc)
endif()

if(WITH_NCCL OR WITH_RCCL)
  list(APPEND COLLECTIVE_SRCS process_group_nccl.cc)
  list(APPEND COLLECTIVE_SRCS async_load.cc)
endif()

if(WITH_XPU_BKCL)
  list(APPEND COLLECTIVE_SRCS process_group_bkcl.cc bkcl_tools.cc)
endif()

if(WITH_MPI)
  list(APPEND COLLECTIVE_SRCS process_group_mpi.cc mpi_tools.cc)
endif()

if(WITH_CUSTOM_DEVICE)
  list(APPEND COLLECTIVE_SRCS process_group_custom.cc custom_ccl_tools.cc)
endif()

list(APPEND COLLECTIVE_SRCS processgroup_comm_utils.cc)

collect_srcs(core_srcs SRCS ${COLLECTIVE_SRCS})
