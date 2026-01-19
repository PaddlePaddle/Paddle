cc_library(
  code_generator
  SRCS operation.cc code_generator.cc code_generator_helper.cc
  DEPS graph subgraph_detector)

cc_library(
  fusion_group_pass
  SRCS fusion_group_pass.cc elementwise_group_detector.cc
  DEPS subgraph_detector fuse_pass_base code_generator phi common)
