cc_library(
  tcp_store
  SRCS tcp_store.cc tcp_utils.cc socket.cpp store.cc
  DEPS enforce glog)

if(WITH_GLOO)
  cc_library(
    gloo_store
    SRCS gloo_store.cc
    DEPS gloo)
endif()

if(NOT WIN32)
  cc_test(
    test_c_tcp_store
    SRCS test_tcp_store.cc
    DEPS tcp_store)
endif()
