cc_library(dygraph_node SRCS nodes.cc DEPS ${eager_deps} ${fluid_deps})
add_dependencies(dygraph_node eager_codegen)