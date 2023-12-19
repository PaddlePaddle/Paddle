core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  dim.cc
  domain.cc
  domain_add_unit_loop_mutator.cc
  map.cc
  stage.cc
  isl_utils.cc
  schedule.cc
  naive_scheduler.cc
  poly_scheduler.cc
  ast_gen.cc
  graph.cc
  compute_at_transform.cc)

# cinn_cc_test(test_graph SRCS graph_test.cc DEPS core)
cinn_cc_test(test_schedule SRCS schedule_test.cc DEPS cinncore)
# cinn_cc_test(test_poly_scheduler SRCS poly_scheduler_test.cc DEPS core)
cinn_cc_test(test_stage SRCS stage_test.cc DEPS cinncore)
cinn_cc_test(test_compute_at_transform SRCS compute_at_transform_test.cc DEPS
             cinncore)
cinn_cc_test(test_ast_gen SRCS ast_gen_test.cc DEPS cinncore)
cinn_cc_test(test_isl_utils SRCS isl_utils_test.cc DEPS cinncore)
