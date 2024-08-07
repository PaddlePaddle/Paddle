# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.

# The following groups are to make better use of `/MP` which MSVC's parallel
# compilation instruction when compiling in Unity Build.
register_unity_group(cu frobenius_norm_op.cu)
register_unity_group(cu logsumexp_op.cu)
