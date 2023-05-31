core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  tensor.cc
  scope.cc
  variable.cc
  buffer.cc
  memory.cc
  instruction.cc
  parallel_compiler.cc
  graph_compiler.cc
  graph.cc
  node.cc
  pass.cc
  op_strategy.cc
  op_lowering.cc
  op_lowering_util.cc
  accuracy_checker.cc
  visualize_helper.cc)

if(WITH_CUDA)
  nv_test(
    test_hlir_framework_buffer
    SRCS buffer_test.cc
    DEPS cinncore)
  cc_test(
    test_hlir_framework_accuracy_checker
    SRCS accuracy_checker_test.cc
    DEPS cinncore)
  cc_test(
    test_hlir_framework_parallel_compiler
    SRCS parallel_compiler_test.cc
    DEPS cinncore)
else()
  cc_test(
    test_hlir_framework_buffer
    SRCS buffer_test.cc
    DEPS cinncore)
endif()

if(WITH_CUDA)
  cc_test(
    test_hlir_framework_op_lowering
    SRCS op_lowering_test.cc
    DEPS cinncore decomposer_test_helper)
endif()
cc_test(
  test_hlir_framework_tensor
  SRCS tensor_test.cc
  DEPS cinncore)
cc_test(
  test_hlir_framework_scope
  SRCS scope_test.cc
  DEPS cinncore)
cc_test(
  test_hlir_framework_instruction
  SRCS instruction_test.cc
  DEPS cinncore)
cc_test(
  test_hlir_framework_op
  SRCS op_test.cc
  DEPS cinncore)
cc_test(
  test_hlir_framework_print_graph_pass
  SRCS print_graph_pass_test.cc
  DEPS cinncore)
cc_test(
  test_hlir_framework_graph
  SRCS graph_test.cc
  DEPS cinncore)

#cc_test(test_hlir_framework_graph_compiler SRCS graph_compiler_test.cc DEPS cinncore)
