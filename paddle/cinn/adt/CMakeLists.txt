core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  equation_solver.cc
  equation_value.cc
  generate_map_expr.cc
  kgroup.cc
  m_ir.cc
  naive_op_equation_context.cc
  partition_op_stmts.cc
  print_map_expr.cc
  schedule_descriptor.cc
  simplify_value.cc)

cinn_cc_test(equation_value_match_trait_test SRCS
             equation_value_match_trait_test.cc DEPS gtest glog)

cinn_cc_test(tree_test SRCS tree_test.cc DEPS gtest glog)

message(STATUS "ADT srcs: ${cinnapi_src}")
