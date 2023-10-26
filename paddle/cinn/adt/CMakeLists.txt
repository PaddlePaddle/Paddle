core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  anchor_sd_equation_context.cc
  equation_function.cc
  equation_solver.cc
  equation_value.cc
  generate_map_expr.cc
  get_sub_reshape_dim_ranges.cc
  igroup.cc
  index_expr_infer_context.cc
  kgroup.cc
  m_ir.cc
  naive_bidirection_equation_generator.cc
  naive_op_equation_context.cc
  partition_op_stmts.cc
  print_constant.cc
  print_equations.cc
  print_map_expr.cc
  print_value.cc
  schedule_descriptor.cc
  schedule_dim.cc
  schedule_mesh.cc
  simplify_value.cc
  print_equations.cc
  print_constant.cc
  print_value.cc
  print_schedule_descriptor.cc
  write_broadcast_disabled_bidirection_equation_generator.cc
  print_schedule_mesh.cc
  print_schedule_dim.cc
  print_loop_size.cc)

cinn_cc_test(equation_value_match_trait_test SRCS
             equation_value_match_trait_test.cc DEPS gtest glog)

cinn_cc_test(tree_test SRCS tree_test.cc DEPS gtest glog)

cinn_cc_test(inline_translator_test SRCS inline_translator_test.cc DEPS
             cinncore)

message(STATUS "ADT srcs: ${cinnapi_src}")
