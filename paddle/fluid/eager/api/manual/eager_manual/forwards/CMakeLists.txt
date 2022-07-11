cc_library(
  conv2d_fwd_function
  SRCS conv2d_fwd_function.cc
  DEPS ${eager_deps} ${fluid_deps} ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})

add_dependencies(conv2d_fwd_function eager_codegen)

set(eager_manual_functions
    conv2d_fwd_function
    PARENT_SCOPE)
