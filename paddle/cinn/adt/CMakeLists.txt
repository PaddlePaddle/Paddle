core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  equation_solver.cc
  generate_map_expr.cc
  kgroup.cc
  m_ir.cc
  naive_op_equation_context.cc
  partition_op_stmts.cc
  print_map_expr.cc
  simplify_value.cc)

message(STATUS "ADT srcs: ${cinnapi_src}")
