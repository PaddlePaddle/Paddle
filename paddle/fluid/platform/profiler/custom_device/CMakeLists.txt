if(WITH_CUSTOM_DEVICE)
  cc_library(
    custom_tracer
    SRCS custom_tracer.cc
    DEPS workqueue_utils enforce glog phi_backends)
else()
  cc_library(
    custom_tracer
    SRCS custom_tracer.cc
    DEPS workqueue_utils enforce glog)
endif()
