core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  replace_call_with_expr.cc
  replace_var_with_expr.cc
  ir_simplify.cc
  optimize.cc
  vectorize_loops.cc
  unroll_loops.cc
  transform_polyfor_to_for.cc
  eliminate_broadcast_in_forloop.cc
  fold_cinn_call_arguments.cc
  call_arg_list_to_pod_value.cc
  insert_debug_log_callee.cc
  lower_function_call_bind_vars.cc
  extern_call_process.cc
  map_extern_call.cc
  compute_inline_expand.cc
  buffer_assign.cc
  replace_const_param_to_integer.cc
  lower_intrin.cc
  cast_bool_to_int8.cc
  var_mod_simplify.cc
  remove_schedule_block.cc
  replace_cross_thread_reduction.cc)

if(WITH_CUDA)
  gather_srcs(cinnapi_src SRCS transform_gpu_forloop.cc)
endif()

cinn_cc_test(test_ir_simplify SRCS ir_simplify_test.cc DEPS cinncore)
cinn_cc_test(test_replace_call_with_expr SRCS replace_call_with_expr_test.cc
             DEPS cinncore)
cinn_cc_test(
  test_vectorize_loops
  SRCS
  vectorize_loops_test.cc
  DEPS
  cinncore
  ARGS
  ${global_test_args})
cinn_cc_test(
  test_transform_polyfor_to_for
  SRCS
  transform_polyfor_to_for_test.cc
  DEPS
  cinncore
  ARGS
  ${global_test_args})
cinn_cc_test(test_optimize SRCS optimize_test.cc DEPS cinncore)
cinn_cc_test(test_cache_read_write_replace SRCS
             cache_read_write_replace_test.cc DEPS cinncore)
cinn_cc_test(test_cast_simplify SRCS cast_simplify_test.cc DEPS cinncore)
cinn_cc_test(test_remove_schedule_block SRCS remove_schedule_block_test.cc DEPS
             cinncore)
cinn_cc_test(test_unroll_loops SRCS unroll_loops_test.cc DEPS cinncore)
cinn_cc_test(test_replace_cross_thread_reduction SRCS
             replace_cross_thread_reduction_test.cc DEPS cinncore)
