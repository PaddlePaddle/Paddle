cc_library(eager_scale SRCS scale.cc DEPS phi_api phi autograd_meta scale_node)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
cc_library(final_dygraph_function SRCS dygraph_functions.cc DEPS ${eager_deps})
add_dependencies(final_dygraph_function eager_final_state_codegen)
endif()
