# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(cc
    reduce_all_op.cc
    reduce_any_op.cc
    reduce_prod_op.cc
    reduce_sum_op.cc)
register_unity_group(cu
    reduce_all_op.cu
    reduce_any_op.cu
    reduce_prod_op.cu
    reduce_prod_op.part.cu
    reduce_sum_op.cu
    reduce_sum_op.part.cu)
# The following groups are to make better use of `/MP` which MSVC's parallel
# compilation instruction when compiling in Unity Build.
register_unity_group(cu frobenius_norm_op.cu)
register_unity_group(cu logsumexp_op.cu)
register_unity_group(cu reduce_max_op.cu)
register_unity_group(cu reduce_min_op.cu)
