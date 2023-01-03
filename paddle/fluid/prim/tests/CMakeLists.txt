cc_test(
  test_static_prim
  SRCS test_static_prim.cc
  DEPS static_utils static_prim_api generated_op prim_utils)
cc_test(
  test_eager_prim
  SRCS test_eager_prim.cc
  DEPS ${eager_deps} ${generated_deps})
