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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
endif()
