include_directories(${CMAKE_SOURCE_DIR}/paddle/cinn/runtime)
set(srcs test_utils.cc test_matmul.cc test_elementwise.cc
         test_all_ops_default.cc)

#cinn_cc_test(test_bk_matmul SRCS test_matmul.cc test_utils.cc DEPS cinncore ARGS ${global_test_args})
#target_compile_options(test_bk_matmul PRIVATE "-O3")

cinn_cc_test(
  test_bk_elementwise
  SRCS
  test_elementwise.cc
  test_utils.cc
  DEPS
  cinncore
  ARGS
  ${global_test_args})
target_compile_options(test_bk_elementwise PRIVATE "-O3")

#cinn_cc_test(test_all_ops_default SRCS test_all_ops_default.cc test_utils.cc DEPS cinncore ARGS ${global_test_args})
#target_compile_options(test_all_ops_default PRIVATE "-O3")
