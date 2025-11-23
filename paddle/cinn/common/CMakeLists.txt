core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  shared.cc
  cinn_value.cc
  type.cc
  target.cc
  arch_util.cc
  object.cc
  debug_manager.cc
  info_registry.cc
  graph_utils.cc
  context.cc
  axis.cc
  ir_util.cc
  test_helper.cc
  # cuda_test_helper.cc
  union_find.cc
  nvgpu_dev_info.cc
  integer_set.cc
  dim_expr_converter.cc
  broadcast_tree.cc
  iter_simplify.cc
  simplify_special_pattern.cc
  shape_constraint.cc)
