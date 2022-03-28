core_gather_headers()

gather_srcs(infrt_src SRCS
    core_op_fuse_pass.cc
    )

mlir_add_rewriter(core_op_fuse)
