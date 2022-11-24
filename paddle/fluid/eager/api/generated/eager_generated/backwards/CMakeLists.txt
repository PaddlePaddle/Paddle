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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
endif()
