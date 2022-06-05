# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(cc accuracy_op.cc auc_op.cc precision_recall_op.cc)
register_unity_group(cu accuracy_op.cu auc_op.cu)
