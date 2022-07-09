cc_library(
  add_n_node
  SRCS add_n_node.cc
  DEPS ${eager_deps} ${fluid_deps})

cc_library(
  conv2d_nodes
  SRCS conv2d_nodes.cc
  DEPS ${eager_deps} ${fluid_deps})

set(eager_manual_nodes
    conv2d_nodes add_n_node
    PARENT_SCOPE)
