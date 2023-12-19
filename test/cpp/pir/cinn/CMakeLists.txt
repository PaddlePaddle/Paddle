add_subdirectory(adt)

if(WITH_TESTING AND WITH_CINN)
  paddle_test(test_pir_compiler SRCS pir_compiler_test.cc DEPS pir_compiler
              cinn_runtime_dialect)

  paddle_test(test_jit_instruction SRCS jit_instruction_test.cc DEPS
              cinn_runtime_dialect pir_compiler)

  paddle_test(
    test_dialect_convert
    SRCS
    dialect_convert_test.cc
    DEPS
    drr
    pd_to_cinn_pass
    op_dialect_vjp
    cinn_op_dialect
    pir)

  paddle_test(
    test_add_broadcast_to_elementwise
    SRCS
    add_broadcast_to_elementwise_test.cc
    DEPS
    drr
    pd_to_cinn_pass
    op_dialect_vjp
    cinn_op_dialect
    add_broadcast_to_elementwise_pass
    pir)

  paddle_test(
    test_sub_graph_extract
    SRCS
    sub_graph_extract_test.cc
    DEPS
    drr
    pd_to_cinn_pass
    cinn_op_dialect
    op_dialect_vjp
    pir_transforms
    pir)

  paddle_test(
    test_ir_op_fusion
    SRCS
    ir_op_fusion_test.cc
    DEPS
    op_with_group_merge_pass
    cinn_op_dialect
    pir)

  paddle_test(
    test_pir_all_path
    SRCS
    pir_all_path_test.cc
    DEPS
    op_with_group_merge_pass
    pir_transforms
    cinn_op_dialect
    pd_to_cinn_pass
    add_broadcast_to_elementwise_pass)

  paddle_test(
    test_group_op
    SRCS
    group_op_test.cc
    DEPS
    pd_to_cinn_pass
    add_broadcast_to_elementwise_pass
    op_with_group_merge_pass
    cinn_op_dialect
    pir_transforms)

  paddle_test(test_pir_build_cinn_pass SRCS build_cinn_pass_test.cc DEPS
              pir_transforms pir)

  paddle_test(test_compilation_task SRCS compilation_task_test.cc DEPS pir)

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
      test_compilation_task)

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
