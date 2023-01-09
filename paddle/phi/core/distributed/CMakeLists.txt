add_subdirectory(store)

set(COMM_CONTEXT_MANAGER_DEPS tcp_store)

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    nccl_comm_context
    SRCS nccl_comm_context.cc
    DEPS dense_tensor)
  list(APPEND COMM_CONTEXT_MANAGER_DEPS nccl_comm_context)
endif()

cc_library(
  comm_context_manager
  SRCS comm_context_manager.cc
  DEPS ${COMM_CONTEXT_MANAGER_DEPS})
