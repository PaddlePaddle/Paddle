cc_library(
  tcp_store
  SRCS tcp_store.cc tcp_utils.cc socket.cpp
  DEPS enforce glog)
