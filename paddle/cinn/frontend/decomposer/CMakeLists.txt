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

if(WITH_CUDA)
  cinn_cc_test(test_activation_decomposer SRCS activation_test.cc DEPS cinncore
               decomposer_test_helper)
  cinn_cc_test(test_elementwise_decomposer SRCS elementwise_test.cc DEPS
               cinncore decomposer_test_helper)
  cinn_cc_test(test_broadcast_decomposer SRCS broadcast_test.cc DEPS cinncore
               decomposer_test_helper)
  cinn_cc_test(test_batch_norm_decomposer SRCS batch_norm_test.cc DEPS cinncore
               decomposer_test_helper)
  cinn_cc_test(test_top_k_decomposer SRCS top_k_test.cc DEPS cinncore
               decomposer_test_helper)
endif()
