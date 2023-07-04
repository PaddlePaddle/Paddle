core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  ir.cc
  ir_base.cc
  function_definition.cc
  ir_operators.cc
  buffer.cc
  function_base.cc
  operation.cc
  registry.cc
  tensor.cc
  module.cc
  lowered_func.cc
  intrinsic_ops.cc
  layout.cc)

add_subdirectory(test)
add_subdirectory(utils)
add_subdirectory(schedule)
