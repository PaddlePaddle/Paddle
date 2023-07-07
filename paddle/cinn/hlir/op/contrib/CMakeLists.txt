core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  gather_nd.cc
  sort.cc
  argmin.cc
  argmax.cc
  repeat.cc
  lookup_table.cc
  one_hot.cc
  reciprocal.cc
  gaussian_random.cc
  uniform_random.cc
  cholesky.cc
  triangular_solve.cc
  bitcast_convert.cc
  randint.cc
  resize.cc
  assert_true.cc)

cinn_cc_test(test_gather_nd SRCS gather_nd_test.cc DEPS cinncore)
cinn_cc_test(test_sort SRCS sort_test.cc DEPS cinncore)
cinn_cc_test(test_argmin SRCS argmin_test.cc DEPS cinncore)
cinn_cc_test(test_argmax SRCS argmax_test.cc DEPS cinncore)
cinn_cc_test(test_repeat SRCS repeat_test.cc DEPS cinncore)
cinn_cc_test(test_one_hot SRCS one_hot_test.cc DEPS cinncore)
cinn_cc_test(test_lookup_table SRCS lookup_table_test.cc DEPS cinncore)
cinn_cc_test(test_reciprocal SRCS reciprocal_test.cc DEPS cinncore)
