core_gather_headers()

gather_srcs(infrt_src SRCS
    infrt_dialect.cc
    )

add_mlir_dialect(infrt_ops infrt)
