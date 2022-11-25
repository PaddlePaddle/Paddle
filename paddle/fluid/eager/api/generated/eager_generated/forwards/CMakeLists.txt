cc_library(
  eager_scale
  SRCS scale.cc
  DEPS phi_api phi autograd_meta scale_node)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    final_dygraph_function
    SRCS dygraph_functions.cc ${eager_manual_functions}
    DEPS ${eager_deps})
<<<<<<< HEAD
  add_dependencies(final_dygraph_function eager_codegen)
  target_link_libraries(final_dygraph_function ${PYTHON_LIBRARIES})
=======
  add_dependencies(final_dygraph_function eager_final_state_codegen)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
endif()
