cc_library(
  global_utils
  SRCS global_utils.cc
  DEPS phi common tracer)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    hook_utils
    SRCS hook_utils.cc
    DEPS phi common autograd_meta grad_node_info utils accumulation_node)
else()
  cc_library(
    hook_utils
    SRCS hook_utils.cc
    DEPS phi common autograd_meta grad_node_info utils)
endif()
