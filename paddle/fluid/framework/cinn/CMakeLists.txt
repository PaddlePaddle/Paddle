cc_library(cinn_cache_key SRCS cinn_cache_key.cc DEPS boost graph graph_helper lod_tensor proto_desc)
cc_library(cinn_compiled_object SRCS cinn_compiled_object.cc DEPS feed_fetch_method graph lod_tensor proto_desc)
cc_library(cinn_runner SRCS cinn_runner.cc DEPS cinn_cache_key cinn_compiled_object feed_fetch_method graph lod_tensor scope)

cc_test(cinn_cache_key_test SRCS cinn_cache_key_test.cc DEPS cinn_cache_key)
cc_test(cinn_runner_test SRCS cinn_runner_test.cc DEPS cinn_runner proto_desc)
cc_test(cinn_compiled_object_test SRCS cinn_compiled_object_test.cc DEPS cinn_compiled_object)
