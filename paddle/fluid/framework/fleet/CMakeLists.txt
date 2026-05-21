cc_library(
  fleet_wrapper
  SRCS fleet_wrapper.cc
  DEPS framework_proto variable_helper scope)

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    nccl_wrapper
    SRCS nccl_wrapper.cc
    DEPS framework_proto variable_helper scope)
endif()

if(WITH_GLOO)
  cc_library(
    gloo_wrapper
    SRCS gloo_wrapper.cc
    DEPS framework_proto variable_helper scope gloo)
else()
  cc_library(
    gloo_wrapper
    SRCS gloo_wrapper.cc
    DEPS framework_proto variable_helper scope)
endif()
