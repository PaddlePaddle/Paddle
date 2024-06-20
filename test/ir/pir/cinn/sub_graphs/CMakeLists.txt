if(WITH_GPU)
  set(STATIC_BUILD_TESTS
      test_sub_graph_0
      test_sub_graph_1
      test_sub_graph_2
      test_sub_graph_3
      test_sub_graph_5
      test_sub_graph_10
      test_sub_graph_12
      test_sub_graph_13
      test_sub_graph_16
      test_sub_graph_17
      test_sub_graph_18
      test_sub_graph_22
      test_sub_graph_23
      test_sub_graph_24
      test_sub_graph_25
      test_sub_graph_26
      test_sub_graph_27
      test_sub_graph_28
      test_sub_graph_29
      test_sub_graph_30
      test_sub_graph_31
      test_sub_graph_33
      test_sub_graph_34
      test_sub_graph_40
      test_sub_graph_42
      test_sub_graph_43
      test_sub_graph_44
      test_sub_graph_48
      test_sub_graph_49
      test_sub_graph_add_n
      test_sub_graph_add
      test_sub_graph_avg_pool2d
      test_sub_graph_chunk
      test_sub_graph_max_pool2d
      test_sub_graph_reshape
      test_sub_graph_swish
      test_sub_graph_mul_method
      test_sub_graph_adaptive_avg_pool2d
      test_sub_graph_52
      test_sub_graph_55
      test_sub_graph_56
      test_sub_graph_59
      test_sub_graph_61
      test_sub_graph_62
      test_sub_graph_63
      test_sub_graph_66
      test_sub_graph_67
      test_sub_graph_69
      test_sub_graph_73
      test_sub_graph_74
      test_sub_graph_77
      test_sub_graph_79
      test_sub_graph_80
      test_sub_graph_82
      test_sub_graph_84
      test_sub_graph_86
      test_sub_graph_87
      test_sub_graph_88)

  file(
    GLOB DYNAMIC_BUILD_TESTS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" DYNAMIC_BUILD_TESTS "${DYNAMIC_BUILD_TESTS}")
  #list(REMOVE_ITEM DYNAMIC_BUILD_TESTS STATIC_BUILD_TESTS)

  foreach(static_test ${STATIC_BUILD_TESTS})
    list(REMOVE_ITEM DYNAMIC_BUILD_TESTS ${static_test})
  endforeach()

  foreach(cinn_sub_graph_test_name ${STATIC_BUILD_TESTS})
    add_test(
      NAME ${cinn_sub_graph_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_cinn_new_group_scheduler=1 FLAGS_enable_pir_api=1
        FLAGS_cinn_bucket_compile=1 FLAGS_group_schedule_tiling_first=1
        FLAGS_cudnn_deterministic=true ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_sub_graph_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES LABELS
                                                                "RUN_TYPE=CINN")
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES TIMEOUT 300)
  endforeach()

  foreach(cinn_sub_graph_test_name ${DYNAMIC_BUILD_TESTS})
    add_test(
      NAME ${cinn_sub_graph_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_prim_enable_dynamic=1 FLAGS_cinn_new_group_scheduler=1
        FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1
        FLAGS_cinn_bucket_compile=1 FLAGS_group_schedule_tiling_first=1
        FLAGS_cudnn_deterministic=true ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_sub_graph_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES LABELS
                                                                "RUN_TYPE=CINN")
    set_tests_properties(${cinn_sub_graph_test_name} PROPERTIES TIMEOUT 600)
  endforeach()

  # set_tests_properties(test_sub_graph_3 PROPERTIES TIMEOUT 300)
  # set_tests_properties(test_sub_graph_54 PROPERTIES TIMEOUT 300)
  # set_tests_properties(test_sub_graph_30 PROPERTIES TIMEOUT 300)

endif()
