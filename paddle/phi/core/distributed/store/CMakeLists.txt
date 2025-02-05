set(STORE_COMMON_SRCS tcp_store.cc tcp_utils.cc socket.cpp store.cc
                      store_utils.cc)

if(WITH_GLOO)
  list(APPEND STORE_COMMON_SRCS gloo_store.cc)
endif()

collect_srcs(core_srcs SRCS ${STORE_COMMON_SRCS})
