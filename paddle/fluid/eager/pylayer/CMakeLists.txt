cc_library(
  py_layer_node
  SRCS py_layer_node.cc
  DEPS pybind phi grad_node_info)
