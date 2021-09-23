cc_library(cinn_runner SRCS cinn_runner.cc DEPS graph scope tensor)
cc_test(cinn_runner_test SRCS cinn_runner_test.cc DEPS cinn_runner proto_desc)
