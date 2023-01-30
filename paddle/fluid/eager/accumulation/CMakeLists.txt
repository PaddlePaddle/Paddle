<<<<<<< HEAD
if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    accumulation_node
    SRCS accumulation_node.cc
    DEPS gradient_accumulator phi_api grad_node_info)
endif()
=======
cc_library(
  accumulation_node
  SRCS accumulation_node.cc
  DEPS gradient_accumulator phi_api grad_node_info)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
