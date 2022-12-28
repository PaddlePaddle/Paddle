cc_test(
  test_static_prim
  SRCS test_static_prim.cc
  DEPS static_utils static_prim_api generated_op)
