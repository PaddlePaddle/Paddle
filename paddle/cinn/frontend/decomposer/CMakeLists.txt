core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  activation.cc
  elementwise.cc
  broadcast.cc
  batch_norm.cc
  top_k.cc)

cinn_cc_library(decomposer_test_helper SRCS test_helper.cc DEPS cinncore)
