# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(
  cc
  ftrl_op.cc
  lars_momentum_op.cc
  proximal_gd_op.cc
  decayed_adagrad_op.cc
  adadelta_op.cc
  dpsgd_op.cc)
register_unity_group(
  cu
  ftrl_op.cu
  lars_momentum_op.cu
  momentum_op.cu
  sgd_op.cu
  adagrad_op.cu
  decayed_adagrad_op.cu
  adadelta_op.cu
  lamb_op.cu)
