cc_library(
  eager_to_static
  SRCS run_program_func.cc run_program_utils.cc run_program_impl.cc
  DEPS pybind phi common grad_node_info)
