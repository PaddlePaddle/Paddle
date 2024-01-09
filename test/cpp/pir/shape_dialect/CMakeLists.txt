paddle_test(symbol_dim_expr_test SRCS symbol_dim_expr_test.cc)

if(WITH_CINN)
  paddle_test(shape_optimization_test SRCS shape_optimization_test.cc)
  set_tests_properties(
    shape_optimization_test PROPERTIES ENVIRONMENT
                                       "FLAGS_enable_pir_in_executor=true")
endif()
