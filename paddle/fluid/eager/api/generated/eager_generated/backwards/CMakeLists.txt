cc_library(scale_node SRCS scale_node.cc DEPS global_utils pten pten_api grad_node_info)

if(NOT ON_INFER)
cc_library(final_dygraph_node SRCS nodes.cc DEPS ${eager_deps})
add_dependencies(final_dygraph_node eager_final_state_codegen)
endif()
