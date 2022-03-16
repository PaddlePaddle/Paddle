core_gather_headers()

gather_srcs(infrt_src SRCS
    proto_arg_map_context.cc
    phi_op_convert_pass.cc
    kernel_op_desc.cc
   )

cc_test(test_kernel_op_desc SRCS kernel_op_desc_test.cc DEPS infrt)
