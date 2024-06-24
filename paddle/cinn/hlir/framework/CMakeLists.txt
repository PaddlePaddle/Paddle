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
  graph_compiler_util.cc
  node.cc
  op_strategy.cc
  compile_error.cc)

cinn_cc_library(pir_compiler SRCS pir_compiler.cc DEPS cinnapi op_dialect_vjp)

if(WITH_CUDA)
  cinn_nv_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS cinncore)
else()
  cinn_cc_test(test_hlir_framework_buffer SRCS buffer_test.cc DEPS cinncore)
endif()
