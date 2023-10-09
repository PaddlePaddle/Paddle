core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  naive_bidirection_equation_generator.cc
  equation_solver.cc
  equation_value.cc
  generate_map_expr.cc
  igroup.cc
  index_expr_infer_context.cc
  kgroup.cc
  m_ir.cc
  naive_op_equation_context.cc
  partition_op_stmts.cc
  print_constant.cc
  print_equations.cc
  print_map_expr.cc
  print_value.cc
  schedule_descriptor.cc
  schedule_dim.cc
  simplify_value.cc
  write_broadcast_disabled_bidirection_equation_generator.cc)

cinn_cc_test(equation_value_match_trait_test SRCS
             equation_value_match_trait_test.cc DEPS gtest glog)

cinn_cc_test(tree_test SRCS tree_test.cc DEPS gtest glog)

message(STATUS "ADT srcs: ${cinnapi_src}")
