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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
endif()
