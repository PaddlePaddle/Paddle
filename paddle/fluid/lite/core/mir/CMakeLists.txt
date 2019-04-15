cc_library(mir_node SRCS node.cc)
cc_library(mir_ssa_graph SRCS ssa_graph.cc DEPS mir_node)
cc_library(mir_pass SRCS pass.cc DEPS mir_ssa_graph)
cc_library(mir_pass_manager SRCS pass_manager.cc DEPS mir_pass mir_ssa_graph)
cc_library(mir_pass_registry SRCS pass_registry.cc DEPS mir_pass_manager)
cc_library(mir_demo_pass SRCS demo_pass.cc DEPS mir_pass)

cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS mir_pass_manager mir_demo_pass)
