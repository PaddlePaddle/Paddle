cc_library(graph SRCS graph.cc node)
cc_library(node SRCS node.cc)
cc_library(pass SRCS pass.cc graph node)

cc_test(graph_test SRCS graph_test.cc DEPS graph proto_desc op_registry)
