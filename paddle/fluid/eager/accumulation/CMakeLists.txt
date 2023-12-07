if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    accumulation_node
    SRCS accumulation_node.cc
    DEPS gradient_accumulator phi common grad_node_info final_dygraph_function)
endif()
