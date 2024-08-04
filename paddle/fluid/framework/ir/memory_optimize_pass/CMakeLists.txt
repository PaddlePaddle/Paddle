cc_library(
  conditional_block_op_eager_deletion_pass
  SRCS conditional_block_op_eager_deletion_pass.cc
  DEPS conditional_block_op_helper graph_helper pass)
cc_library(
  pylayer_op_eager_deletion_pass
  SRCS pylayer_op_eager_deletion_pass.cc
  DEPS pylayer_op_helper graph_helper pass)
cc_library(
  while_op_eager_deletion_pass
  SRCS while_op_eager_deletion_pass.cc
  DEPS while_op_helper graph_helper pass)
cc_library(
  reference_count_pass_helper
  SRCS reference_count_pass_helper.cc
  DEPS garbage_collector)

set(EAGER_DELETETION_PASS_DEPS
    graph
    graph_helper
    pass
    conditional_block_op_eager_deletion_pass
    pylayer_op_eager_deletion_pass
    while_op_eager_deletion_pass
    reference_count_pass_helper)

cc_library(
  eager_deletion_pass
  SRCS eager_deletion_pass.cc
  DEPS ${EAGER_DELETETION_PASS_DEPS})
