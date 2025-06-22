cc_library(
  eager_to_static
  SRCS run_program_op_func.cc utils.cc
  DEPS pybind phi common grad_node_info)
