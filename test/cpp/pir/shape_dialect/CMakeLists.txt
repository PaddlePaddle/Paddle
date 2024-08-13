paddle_test(symbol_dim_expr_test SRCS symbol_dim_expr_test.cc)
paddle_test(simplify_dim_expr_test SRCS simplify_dim_expr_test.cc)
paddle_test(dim_expr_util_test SRCS dim_expr_util_test.cc)
paddle_test(constraints_manager_test SRCS constraints_manager_test.cc)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(symbol_dim_expr_test)
endif()
