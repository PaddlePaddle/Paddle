cc_test(
  test_static_prim
  SRCS test_static_prim.cc
  DEPS utils_static static_prim_ops generated_op)
