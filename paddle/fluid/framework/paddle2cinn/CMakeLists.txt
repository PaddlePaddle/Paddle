cc_library(cinn_cache_key SRCS cinn_cache_key.cc DEPS boost graph graph_helper lod_tensor proto_desc)
cc_library(cinn_compiler SRCS cinn_compiler.cc DEPS cinn_cache_key graph lod_tensor scope cinn)
cc_library(build_cinn_pass SRCS build_cinn_pass.cc DEPS pass subgraph_detector)

cc_test(cinn_cache_key_test SRCS cinn_cache_key_test.cc DEPS cinn_cache_key)
cc_test(cinn_compiler_test SRCS cinn_compiler_test.cc DEPS cinn_compiler proto_desc cinn)
cc_test(build_cinn_pass_test SRCS build_cinn_pass_test.cc DEPS build_cinn_pass)
