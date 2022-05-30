core_gather_headers()

gather_srcs(infrt_src SRCS
    pd_ops.cc
    )
add_mlir_dialect(pd_ops pd)
mlir_tablegen_on(pd_extra_ops)
