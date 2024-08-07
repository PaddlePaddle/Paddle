cc_library(
  op_graph_view
  SRCS op_graph_view.cc
  DEPS detail_op_handle)
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
  DEPS garbage_collector) #
cc_library(
  reference_count_pass
  SRCS reference_count_pass.cc
  DEPS graph graph_helper pass op_graph_view reference_count_pass_helper)

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

cc_library(
  memory_reuse_pass
  SRCS memory_reuse_pass.cc
  DEPS reference_count_pass_helper graph pass)

cc_library(
  buffer_shared_inplace_op_pass
  SRCS buffer_shared_inplace_op_pass.cc
  DEPS memory_reuse_pass executor_gc_helper)
cc_library(
  buffer_shared_cross_op_memory_reuse_pass
  SRCS buffer_shared_cross_op_memory_reuse_pass.cc
  DEPS memory_reuse_pass)

cc_library(
  inplace_addto_op_pass
  SRCS inplace_addto_op_pass.cc
  DEPS memory_reuse_pass)
