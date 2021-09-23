cc_library(cinn_cache_key SRCS cinn_cache_key.cc DEPS boost graph graph_helper lod_tensor proto_desc)
cc_library(cinn_runner SRCS cinn_runner.cc DEPS graph lod_tensor scope)

cc_test(cinn_cache_key_test SRCS cinn_cache_key_test.cc DEPS cinn_cache_key)
cc_test(cinn_runner_test SRCS cinn_runner_test.cc DEPS cinn_runner proto_desc)
