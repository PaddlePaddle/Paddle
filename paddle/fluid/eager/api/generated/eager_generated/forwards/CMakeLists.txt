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
  add_dependencies(final_dygraph_function eager_final_state_codegen)
=======
  add_dependencies(final_dygraph_function eager_codegen)
  target_link_libraries(final_dygraph_function ${PYTHON_LIBRARIES})
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
endif()
