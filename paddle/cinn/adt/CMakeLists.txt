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
  simplify_value.cc
  write_broadcast_disabled_bidirection_equation_generator.cc)
