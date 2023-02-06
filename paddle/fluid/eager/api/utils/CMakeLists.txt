cc_library(
  global_utils
  SRCS global_utils.cc
  DEPS place tracer)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    eager_tensor_operants
    SRCS eager_tensor_operants.cc
    DEPS final_dygraph_function)
endif()

cc_library(
  static_tensor_operants
  SRCS static_tensor_operants.cc
  DEPS static_prim_api)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    tensor_utils
    SRCS tensor_utils.cc
    DEPS phi_api autograd_meta grad_node_info accumulation_node)
  cc_library(
    hook_utils
    SRCS hook_utils.cc
    DEPS phi tensor_utils autograd_meta grad_node_info utils accumulation_node)
else()
  cc_library(
    tensor_utils
    SRCS tensor_utils.cc
    DEPS phi_api autograd_meta grad_node_info)
  cc_library(
    hook_utils
    SRCS hook_utils.cc
    DEPS phi tensor_utils autograd_meta grad_node_info utils)
endif()
