cc_library(analysis SRCS dot.cc node.cc)
cc_test(test_node SRCS node_tester.cc DEPS analysis)
