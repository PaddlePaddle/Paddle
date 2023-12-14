if(NOT CINN_ONLY)
  add_subdirectory(print_utils)

  core_gather_headers()

  gather_srcs(
    cinnapi_src
    SRCS
    adapter_tensor.cc
    anchor_sd_equation_context.cc
    equation_function.cc
    equation_solver.cc
    equation_value.cc
    generate_map_expr.cc
    get_sub_reshape_dim_ranges.cc
    graph_symbolic_dim_infer_ctx.cc
    igroup.cc
    index_expr_infer_context.cc
    kgroup.cc
    m_ir.cc
    naive_bidirection_equation_generator.cc
    naive_op_equation_context.cc
    partition_op_stmts.cc
    schedule_descriptor.cc
    schedule_dim.cc
    schedule_mesh.cc
    dim_expr.cc
    dim_expr_simplifier.cc
    simplify_value.cc
    write_broadcast_disabled_bidirection_equation_generator.cc)

  cinn_cc_test(equation_value_match_trait_test SRCS
               equation_value_match_trait_test.cc DEPS gtest glog)

  cinn_cc_test(tree_test SRCS tree_test.cc DEPS gtest glog)

  cinn_cc_test(
    inline_translator_test
    SRCS
    inline_translator_test.cc
    DEPS
    gtest
    glog
    absl)

  cinn_cc_test(
    dim_expr_test
    SRCS
    dim_expr_test.cc
    DEPS
    gtest
    glog
    cinncore)

endif()
