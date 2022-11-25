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
  add_dependencies(final_dygraph_node eager_codegen)
  target_link_libraries(final_dygraph_node ${PYTHON_LIBRARIES})
=======
  add_dependencies(final_dygraph_node eager_final_state_codegen)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
endif()
