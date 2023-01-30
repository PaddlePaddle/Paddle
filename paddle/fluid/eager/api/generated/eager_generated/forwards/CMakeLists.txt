cc_library(
  eager_scale
  SRCS scale.cc
  DEPS phi_api phi autograd_meta scale_node)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    final_dygraph_function
    SRCS dygraph_functions.cc ${eager_manual_functions}
<<<<<<< HEAD
    DEPS ${eager_deps} final_dygraph_node)
=======
    DEPS ${eager_deps})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  add_dependencies(final_dygraph_function eager_codegen)
endif()
