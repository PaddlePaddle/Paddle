cc_library(graph SRCS graph.cc DEPS node)
cc_library(node SRCS node.cc DEPS proto_desc)
cc_library(pass SRCS pass.cc DEPS graph node)

cc_test(graph_test SRCS graph_test.cc DEPS graph proto_desc op_registry)
