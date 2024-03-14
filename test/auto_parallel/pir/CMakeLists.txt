if(WITH_DISTRIBUTE AND WITH_GPU)
  py_test_modules(test_to_static_pir_program MODULES test_to_static_pir_program)
  set_tests_properties(test_to_static_pir_program
                       PROPERTIES ENVIRONMENT "FLAGS_enable_pir_api=1")
endif()
