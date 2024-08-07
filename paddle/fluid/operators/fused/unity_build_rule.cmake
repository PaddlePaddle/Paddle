# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(
  cc
  fused_elemwise_activation_op.cc
  fused_embedding_fc_lstm_op.cc
  fused_embedding_seq_pool_op.cc
  fusion_lstm_op.cc
  fusion_seqpool_concat_op.cc
  multi_gru_op.cc
  onednn/multi_gru_onednn_op.cc
  fusion_seqpool_cvm_concat_op.cc)
