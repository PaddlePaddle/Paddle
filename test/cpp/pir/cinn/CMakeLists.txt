add_subdirectory(adt)

if(WITH_TESTING AND WITH_CINN)
  paddle_test(test_pir_compiler SRCS pir_compiler_test.cc)

  paddle_test(test_jit_instruction SRCS jit_instruction_test.cc)

  paddle_test(test_dialect_convert SRCS dialect_convert_test.cc)

  paddle_test(test_add_broadcast_to_elementwise SRCS
              add_broadcast_to_elementwise_test.cc)

  paddle_test(test_sub_graph_extract SRCS sub_graph_extract_test.cc DEPS
              pir_transforms)

  paddle_test(test_ir_op_fusion SRCS ir_op_fusion_test.cc)

  paddle_test(test_pir_all_path SRCS pir_all_path_test.cc)

  paddle_test(test_group_op SRCS group_op_test.cc)

  paddle_test(test_symbolic_lower SRCS symbolic_lower_test.cc)

  paddle_test(test_pir_build_cinn_pass SRCS build_cinn_pass_test.cc)

  paddle_test(test_compilation_task SRCS compilation_task_test.cc)

  paddle_test(test_generate_shape_util_test SRCS generate_shape_util_test.cc
              DEPS cinn_op_dialect)

  # DO NOT forget add test name here, otherwise it will not be executed in
  # CINN CI.
  set(cinn_unit_tests
      test_pir_compiler
      test_jit_instruction
      test_dialect_convert
      test_add_broadcast_to_elementwise
      test_sub_graph_extract
      test_ir_op_fusion
      test_pir_all_path
      test_group_op
      test_pir_build_cinn_pass
      test_compilation_task
      test_generate_shape_util_test)

  foreach(test_name ${cinn_unit_tests})
    get_property(
      env
      TEST ${test_name}
      PROPERTY ENVIRONMENT)
    set_property(TEST ${test_name}
                 PROPERTY ENVIRONMENT "FLAGS_cinn_new_group_scheduler=1" ${env})
    set_tests_properties(${test_name} PROPERTIES LABELS "RUN_TYPE=CINN")
  endforeach()

endif()
