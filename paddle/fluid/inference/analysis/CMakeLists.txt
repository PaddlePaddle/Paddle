cc_library(analysis SRCS dot.cc node.cc)
cc_test(test_dot SRCS dot_tester.cc DEPS analysis)
cc_test(test_node SRCS node_tester.cc DEPS analysis)
