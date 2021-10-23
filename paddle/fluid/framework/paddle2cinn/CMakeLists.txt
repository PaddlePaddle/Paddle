cc_library(cinn_cache_key SRCS cinn_cache_key.cc DEPS boost graph graph_helper lod_tensor proto_desc)
cc_library(cinn_compiler SRCS cinn_compiler.cc DEPS cinn_cache_key graph lod_tensor scope cinn)
cc_library(build_cinn_pass SRCS build_cinn_pass.cc DEPS pass subgraph_detector cinn_compiler)

if (WITH_CINN)
  cc_library(transform_desc SRCS transform_desc.cc DEPS proto_desc cinn)
  cc_library(cinn_graph_symbolization SRCS cinn_graph_symbolization.cc DEPS lod_tensor graph graph_helper transform_desc cinn)

  cc_test(test_transform_desc SRCS transform_desc_test.cc DEPS transform_desc)
  cc_test(test_cinn_graph_symbolization SRCS cinn_graph_symbolization_test.cc DEPS cinn_graph_symbolization)
endif()

cc_test(cinn_cache_key_test SRCS cinn_cache_key_test.cc DEPS cinn_cache_key)
cc_test(cinn_compiler_test SRCS cinn_compiler_test.cc DEPS cinn_compiler proto_desc cinn graph_viz_pass build_cinn_pass)
cc_test(build_cinn_pass_test SRCS build_cinn_pass_test.cc DEPS build_cinn_pass)
