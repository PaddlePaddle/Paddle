if(NOT WIN32)
  add_subdirectory(nccl)
endif()

collect_srcs(core_srcs SRCS ops_extra_info.cc)
