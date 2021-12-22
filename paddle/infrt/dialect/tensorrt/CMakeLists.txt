core_gather_headers()

gather_srcs(infrt_src SRCS
    trt_ops.cc
    trt_op_teller_pass.cc
    trt_graph_fuse_pass.cc
    )
mlir_tablegen_on(trt_ops)
