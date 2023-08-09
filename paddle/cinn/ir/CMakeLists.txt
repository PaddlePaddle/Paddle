cinn_proto_library(schedule_desc_proto SRCS schedule_desc.proto)

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  ir.cc
  ir_base.cc
  ir_schedule.cc
  ir_schedule_error.cc
  ir_schedule_util.cc
  ir_visitor.cc
  ir_printer.cc
  ir_mutator.cc
  function_definition.cc
  lowered_func.cc
  ir_operators.cc
  ir_verify.cc
  buffer.cc
  function_base.cc
  operation.cc
  collect_ir_nodes.cc
  registry.cc
  tensor.cc
  module.cc
  intrinsic_ops.cc
  layout.cc
  schedule_desc.cc
  ir_compare.cc
  schedule_block_graph.cc)

# cinn_cc_test(test_ir SRCS ir_test.cc DEPS core)
# cinn_cc_test(test_ir_printer SRCS ir_printer_test.cc DEPS core)
# cinn_cc_test(test_ir_operators SRCS ir_operators_test.cc DEPS core)
cinn_cc_test(test_collect_ir_nodes SRCS collect_ir_nodes_test.cc DEPS cinncore)
# cinn_cc_test(test_tensor SRCS tensor_test.cc DEPS core)
cinn_cc_test(
  test_buffer
  SRCS
  buffer_test.cc
  DEPS
  cinncore
  ARGS
  ${global_test_args})
cinn_cc_test(test_tensor SRCS tensor_test.cc DEPS cinncore)
cinn_cc_test(test_intrinsic_ops SRCS intrinsic_ops_test.cc DEPS cinncore)
cinn_cc_test(test_ir_verify SRCS ir_verify_test.cc DEPS cinncore)
cinn_cc_test(test_schedule_desc SRCS schedule_desc_test.cc DEPS cinncore)
cinn_cc_test(test_ir_compare SRCS ir_compare_test.cc DEPS cinncore)
cinn_cc_test(test_schedule_block_graph SRCS schedule_block_graph_test.cc DEPS
             cinncore)

foreach(header ${schedule_desc_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()
