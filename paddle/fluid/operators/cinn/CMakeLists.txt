include(operators)

cc_library(
  cinn_op_helper
  SRCS cinn_op_helper.cc
  DEPS operator device_context)
cc_library(
  cinn_launch_context
  SRCS cinn_launch_context.cc
  DEPS phi
       lod_tensor
       scope
       proto_desc
       graph
       build_strategy
       device_context
       parallel_executor
       standalone_executor
       transform_type
       cinnapi)

set(CINN_OP_DEPS
    parallel_executor
    string_helper
    variable_helper
    cinnapi
    cinn_compiler
    cinn_op_helper
    cinn_launch_context
    transform_type)
register_operators(DEPS ${CINN_OP_DEPS})
