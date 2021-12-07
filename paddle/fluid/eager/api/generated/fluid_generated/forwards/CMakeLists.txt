cc_library(dygraph_function SRCS dygraph_forward_functions.cc DEPS ${eager_deps} ${fluid_deps} ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})
add_dependencies(dygraph_function eager_codegen)