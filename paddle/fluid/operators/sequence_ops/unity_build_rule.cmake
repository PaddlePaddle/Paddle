# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(
  cc
  sequence_conv_op.cc
  sequence_enumerate_op.cc
  sequence_erase_op.cc
  sequence_expand_op.cc
  sequence_mask_op.cc
  sequence_pad_op.cc
  sequence_pool_op.cc
  sequence_expand_as_op.cc
  sequence_reshape_op.cc
  sequence_reverse_op.cc
  sequence_scatter_op.cc
  sequence_slice_op.cc
  sequence_softmax_op.cc
  sequence_topk_avg_pooling_op.cc
  sequence_unpad_op.cc
  sequence_conv_op.cu.cc)
register_unity_group(
  cu
  sequence_enumerate_op.cu
  sequence_erase_op.cu
  sequence_expand_op.cu
  sequence_pad_op.cu
  sequence_expand_as_op.cu
  sequence_reshape_op.cu
  sequence_reverse_op.cu
  sequence_slice_op.cu
  sequence_softmax_cudnn_op.cu.cc
  sequence_softmax_op.cu
  sequence_unpad_op.cu)
