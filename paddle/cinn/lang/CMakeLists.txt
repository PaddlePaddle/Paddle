core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  buffer.cc
  compute.cc
  placeholder.cc
  lower.cc
  lower_impl.cc
  lower_tensor_group.cc
  builtin.cc
  lower_impl.cc
  packed_func.cc)

cinn_cc_test(test_compute SRCS compute_test.cc DEPS cinncore)
cinn_cc_test(test_cinn_packed_func SRCS packed_func_test.cc DEPS cinncore)
