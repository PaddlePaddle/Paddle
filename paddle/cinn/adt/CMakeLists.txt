core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  equation_solver.cc
  generate_map_expr.cc
  simplify_value.cc
  kgroup.cc
  m_ir.cc
  partition_op_stmts.cc)

message(STATUS "ADT srcs: ${cinnapi_src}")
