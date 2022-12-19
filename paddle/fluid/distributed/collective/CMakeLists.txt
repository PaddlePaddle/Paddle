cc_library(
  process_group
  SRCS ProcessGroup.cc
  DEPS dense_tensor)
cc_library(
  process_group_stream
  SRCS process_group_stream.cc
  DEPS dense_tensor)
cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api process_group process_group_stream phi_api string_helper)

if(WITH_DISTRIBUTE)
  cc_library(
    process_group_gloo
    SRCS process_group_gloo.cc
    DEPS phi_api eager_api gloo_wrapper)
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    process_group_nccl
    SRCS process_group_nccl.cc nccl_tools.cc common.cc check.cc
    DEPS process_group
         process_group_stream
         place
         enforce
         collective_helper
         device_context
         ${DEVICE_EVENT_LIBS}
         dense_tensor)
endif()

if(WITH_XPU_BKCL)
  cc_library(
    process_group_bkcl
    SRCS ProcessGroupBKCL.cc BKCLTools.cc common.cc
    DEPS process_group place enforce collective_helper device_context
         dense_tensor)
endif()

if(WITH_MPI)
  cc_library(
    process_group_mpi
    SRCS process_group_mpi.cc mpi_tools.cc common.cc
    DEPS collective_helper device_context)
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
    process_group_custom
    SRCS ProcessGroupCustom.cc CustomCCLTools.cc common.cc
    DEPS process_group phi_backends place enforce collective_helper
         device_context)
endif()
