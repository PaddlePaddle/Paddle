cc_library(
  scale_node
  SRCS scale_node.cc
  DEPS global_utils phi phi_api grad_node_info)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    final_dygraph_node
    SRCS nodes.cc ${eager_manual_nodes}
    DEPS ${eager_deps})
<<<<<<< HEAD
  add_dependencies(final_dygraph_node eager_final_state_codegen)
=======
  add_dependencies(final_dygraph_node eager_codegen)
  target_link_libraries(final_dygraph_node ${PYTHON_LIBRARIES})
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
endif()
