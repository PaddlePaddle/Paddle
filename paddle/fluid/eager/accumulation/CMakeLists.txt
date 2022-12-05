if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    accumulation_node
    SRCS accumulation_node.cc
    DEPS gradient_accumulator phi_api grad_node_info)
  add_dependencies(accumulation_node eager_codegen)
endif()
