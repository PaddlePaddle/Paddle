cc_library(
  conv2d_nodes
  SRCS conv2d_nodes.cc
  DEPS ${eager_deps} ${fluid_deps})

set(eager_manual_nodes
    conv2d_nodes
    PARENT_SCOPE)
