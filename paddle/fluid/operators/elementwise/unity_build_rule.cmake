# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(
  cc
  elementwise_add_op.cc
  elementwise_div_op.cc
  elementwise_floordiv_op.cc
  elementwise_max_op.cc
  elementwise_min_op.cc
  elementwise_mod_op.cc
  elementwise_mul_op.cc
  elementwise_pow_op.cc
  elementwise_sub_op.cc)
register_unity_group(
  cu
  elementwise_add_op.cu
  elementwise_div_op.cu
  elementwise_floordiv_op.cu
  elementwise_max_op.cu
  elementwise_min_op.cu
  elementwise_mod_op.cu
  elementwise_mul_op.cu
  elementwise_pow_op.cu
  elementwise_sub_op.cu)
