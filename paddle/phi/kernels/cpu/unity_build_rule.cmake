# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.

register_unity_group(cc
    sigmoid_cross_entropy_with_logits_grad_kernel.cc
    sigmoid_cross_entropy_with_logits_kernel.cc
    sparse_weight_embedding_grad_kernel.cc
    sparse_weight_embedding_kernel
    truncated_gaussian_random_kernel.cc
    pixel_shuffle_grad_kernel.cc
    searchsorted_kernel.cc
    segment_pool_grad_kernel.cc
    scatter_nd_add_kernel.cc
    scatter_nd_add_grad_kernel.cc
    )
register_unity_group(cc
    cholesky_kernel.cc
    cholesky_grad_kernel.cc
    cholesky_solve_grad_kernel.cc
    cholesky_solve_kernel.cc
    conv_transpose_grad_kernel.cc
    conv_transpose_kernel.cc
    embedding_grad_kernel.cc
    embedding_kernel.cc
    expand_as_grad_kernel.cc
    expand_as_kernel.cc
    frobenius_norm_grad_kernel.cc
    frobenius_norm_kernel.cc
    )