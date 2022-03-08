core_gather_headers()

gather_srcs(infrt_src SRCS
    infrt_op_fuse_pass.cc
    )

mlir_add_rewriter(infrt_op_fuse)
