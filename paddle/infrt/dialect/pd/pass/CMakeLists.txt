add_subdirectory(utils)
core_gather_headers()

gather_srcs(infrt_src SRCS
    conv2d_bn_fuse_utils.cc
    pd_op_fuse_pass.cc
    )

mlir_add_rewriter(pd_op_fuse)
