add_subdirectory(adt)

if(WITH_TESTING AND WITH_CINN)

  paddle_test(test_dialect_convert SRCS dialect_convert_test.cc)

  paddle_test(test_add_broadcast_to_elementwise SRCS
              add_broadcast_to_elementwise_test.cc)

  paddle_test(test_sub_graph_extract SRCS sub_graph_extract_test.cc DEPS
              pir_transforms)

  paddle_test(test_ir_op_cluster SRCS ir_op_cluster_test.cc DEPS pir_transforms
              cinn_transforms)

  paddle_test(test_pir_all_path SRCS pir_all_path_test.cc DEPS cinn_transforms)

  paddle_test(test_symbolic_lower SRCS symbolic_lower_test.cc)

  paddle_test(test_pir_build_cinn_pass SRCS build_cinn_pass_test.cc)

  paddle_test(test_compilation_task SRCS compilation_task_test.cc)

  paddle_test(test_generate_shape_util_test SRCS generate_shape_util_test.cc
              DEPS cinn_op_dialect)

  paddle_test(merge_parallel_matmul_pass_test SRCS
              merge_parallel_matmul_pass_test.cc)

  paddle_test(test_tile_config_searcher SRCS tile_config_searcher_test.cc DEPS
              schedule_config_search)

  paddle_test(
    test_tile_config_searcher_pure_spatial SRCS
    tile_config_searcher_pure_spatial_test.cc DEPS schedule_config_search)

  paddle_test(test_tile_config_performance SRCS tile_config_performance_test.cc
              DEPS schedule_config_search)

  paddle_test(test_file_tile_config SRCS file_tile_config_test.cc)

  paddle_test(replace_cross_block_reduction_test SRCS
              replace_cross_block_reduction_test.cc)

  paddle_test(ir_simplify_test SRCS ir_simplify_test.cc)

  # DO NOT forget add test name here, otherwise it will not be executed in
  # CINN CI.
  set(cinn_unit_tests
      test_dialect_convert
      test_add_broadcast_to_elementwise
      test_sub_graph_extract
      test_pir_all_path
      test_pir_build_cinn_pass
      test_compilation_task
      test_generate_shape_util_test
      merge_parallel_matmul_pass_test
      test_tile_config_searcher
      test_tile_config_searcher_pure_spatial
      test_file_tile_config
      replace_cross_block_reduction_test)

  foreach(test_name ${cinn_unit_tests})
    get_property(
      env
      TEST ${test_name}
      PROPERTY ENVIRONMENT)
    set_property(TEST ${test_name} PROPERTY ENVIRONMENT ${env})
    set_tests_properties(${test_name} PROPERTIES LABELS "RUN_TYPE=CINN")
  endforeach()

endif()
