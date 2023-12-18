cc_library(
  custom_tracer
  SRCS custom_tracer.cc
  DEPS enforce glog workqueue_utils common)
