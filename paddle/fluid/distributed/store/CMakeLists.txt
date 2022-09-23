cc_library(
  tcp_store
  SRCS tcp_store.cc tcp_utils.cc socket.cpp
  DEPS enforce glog)

if(NOT WIN32)
  cc_test(
    test_c_tcp_store
    SRCS test_tcp_store.cc
    DEPS tcp_store)
endif()
