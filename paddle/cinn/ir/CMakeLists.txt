core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  ir.cc
  ir_base.cc
  function_definition.cc
  buffer.cc
  function_base.cc
  operation.cc
  registry.cc
  tensor.cc
  module.cc
  lowered_func.cc
  intrinsic_ops.cc
  layout.cc
  schedule_block_graph.cc)

add_subdirectory(op)
add_subdirectory(test)
add_subdirectory(utils)
add_subdirectory(schedule)
