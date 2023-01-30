cc_library(
  scale_node
  SRCS scale_node.cc
  DEPS global_utils phi phi_api grad_node_info)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    final_dygraph_node
    SRCS nodes.cc ${eager_manual_nodes}
<<<<<<< HEAD
    DEPS ${eager_deps} eager_prim_api)
=======
    DEPS ${eager_deps})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  add_dependencies(final_dygraph_node eager_codegen)
endif()
