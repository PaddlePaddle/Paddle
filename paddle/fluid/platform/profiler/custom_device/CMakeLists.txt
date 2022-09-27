cc_library(
  custom_tracer
  SRCS custom_tracer.cc
  DEPS workqueue_utils enforce glog)
