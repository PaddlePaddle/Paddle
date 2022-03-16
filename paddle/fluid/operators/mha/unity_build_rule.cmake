# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(cc
    mha_data_prepare_op.cc
    mha_op.cc)
register_unity_group(cu
    mha_data_prepare_op.cu
    mha_op.cu)
