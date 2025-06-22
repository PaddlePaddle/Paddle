cc_library(
  eager_to_static_impl
  SRCS run_program_impl.cc run_program_utils.cc
  DEPS phi common pir executor executor_cache)

cc_library(
  eager_to_static_node
  SRCS run_program_node.cc
  DEPS grad_node_info eager_to_static_impl)

cc_library(
  eager_to_static
  SRCS run_program_func.cc
  DEPS pybind eager_to_static_node)
