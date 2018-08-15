cc_library(node SRCS node.cc DEPS proto_desc)
cc_library(graph SRCS graph.cc DEPS node)
cc_library(graph_helper SRCS graph_helper.cc DEPS graph)
cc_library(pass SRCS pass.cc DEPS graph node graph_helper)
cc_library(graph_viz_pass SRCS graph_viz_pass.cc DEPS graph pass graph_helper)

cc_test(pass_test SRCS pass_test.cc DEPS graph pass graph_helper)
cc_test(graph_test SRCS graph_test.cc DEPS graph graph_helper op_registry)
cc_test(graph_helper_test SRCS graph_helper_test.cc DEPS graph graph_helper op_registry)
