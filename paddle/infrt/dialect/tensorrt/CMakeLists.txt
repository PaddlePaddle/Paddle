core_gather_headers()

gather_srcs(infrt_src SRCS
    trt_ops.cc
    trt_op_converter_pass.cc
    trt_op_teller_pass.cc
    trt_graph_fuse_pass.cc
    trt_graph_split_pass.cc
    trt_type_convert_pass.cc
    )
mlir_tablegen_on(trt_ops)
mlir_add_rewriter(pd_lower_to_trt)

add_executable(trt-exec trt_exec.cc)
target_link_libraries(trt-exec infrt ${MLIR_IR_LIBS})
