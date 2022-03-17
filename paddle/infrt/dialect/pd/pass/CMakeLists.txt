
core_gather_headers()

gather_srcs(infrt_src SRCS
    pd_op_fuse_pass.cc
    )

mlir_add_rewriter(pd_op_fuse)
