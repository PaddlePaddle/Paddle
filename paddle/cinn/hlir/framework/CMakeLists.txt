add_subdirectory(pir)
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
  program.cc
  parallel_compiler.cc
  graph_compiler.cc
  graph_compiler_util.cc
  graph.cc
  node.cc
  pass.cc
  op_strategy.cc
  op_lowering_util.cc
  op_lowering_impl.cc
  accuracy_checker.cc
  visualize_helper.cc
  compile_error.cc)

# TODO(Aurelius84): pir_compiler depends on pd_op_dialect and could
# not found under CINN_ONLY mode
if(NOT CINN_ONLY)
  cinn_cc_library(pir_compiler SRCS pir_compiler.cc DEPS cinnapi pd_op_dialect)
endif()

if(WITH_CUDA)
  cinn_nv_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS cinncore)
  cinn_cc_test(test_hlir_framework_accuracy_checker SRCS
               accuracy_checker_test.cc DEPS cinncore)
  cinn_cc_test(test_hlir_framework_parallel_compiler SRCS
               parallel_compiler_test.cc DEPS cinncore)
else()
  cinn_cc_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS cinncore)
endif()

if(WITH_CUDA)
  cinn_cc_test(test_hlir_framework_op_lowering SRCS op_lowering_test.cc DEPS
               cinncore decomposer_test_helper)
endif()
cinn_cc_test(test_hlir_framework_tensor SRCS tensor_test.cc DEPS cinncore)
cinn_cc_test(test_hlir_framework_scope SRCS scope_test.cc DEPS cinncore)
cinn_cc_test(test_hlir_framework_instruction SRCS instruction_test.cc DEPS
             cinncore)
cinn_cc_test(test_hlir_framework_op SRCS op_test.cc DEPS cinncore)
cinn_cc_test(test_hlir_framework_print_graph_pass SRCS print_graph_pass_test.cc
             DEPS cinncore)
cinn_cc_test(test_hlir_framework_graph SRCS graph_test.cc DEPS cinncore)
cinn_cc_test(test_hlir_framework_graph_compiler SRCS graph_compiler_test.cc
             DEPS cinncore)
