add_subdirectory(store)

set(COMM_CONTEXT_MANAGER_DEPS tcp_store)

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    nccl_comm_context
    SRCS nccl_comm_context.cc
    DEPS dense_tensor)
  list(APPEND COMM_CONTEXT_MANAGER_DEPS nccl_comm_context)
endif()

if(WITH_GLOO)
  cc_library(
    gloo_utils
    SRCS gloo_utils.cc
    DEPS gloo dense_tensor enforce)

  cc_library(
    gloo_comm_context
    SRCS gloo_comm_context.cc
    DEPS gloo_utils)

  list(APPEND COMM_CONTEXT_MANAGER_DEPS gloo_comm_context gloo_store)
endif()

cc_library(
  comm_context_manager
  SRCS comm_context_manager.cc
  DEPS ${COMM_CONTEXT_MANAGER_DEPS})
