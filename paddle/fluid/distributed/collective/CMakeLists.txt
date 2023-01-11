cc_library(
  process_group
  SRCS process_group.cc
  DEPS dense_tensor)

cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api process_group phi_api string_helper)

if(WITH_DISTRIBUTE)
  cc_library(
    process_group_gloo
    SRCS process_group_gloo.cc
    DEPS phi_api eager_api gloo_wrapper tcp_store)
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    process_group_nccl
    SRCS process_group_nccl.cc nccl_tools.cc common.cc check.cc
    DEPS process_group
         tcp_store
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
    SRCS process_group_bkcl.cc bkcl_tools.cc common.cc
    DEPS process_group
         tcp_store
         place
         enforce
         collective_helper
         device_context
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
    SRCS process_group_custom.cc custom_ccl_tools.cc common.cc
    DEPS process_group
         tcp_store
         phi_backends
         place
         enforce
         collective_helper
         device_context)
endif()
