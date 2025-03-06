cinn_cc_test(test_dfs_walker SRCS dfs_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_dfs_topo_walker SRCS dfs_topo_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_cinn_value SRCS cinn_value_test.cc DEPS cinncore)
cinn_cc_test(test_axis SRCS axis_test.cc DEPS cinncore)

cinn_cc_test(dim_expr_converter_test SRCS dim_expr_converter_test.cc DEPS
             cinncore)
cinn_cc_test(broadcast_tree_test SRCS broadcast_tree_test.cc DEPS cinncore)

cinn_cc_test(test_equation_graph_topo_walker SRCS
             equation_graph_topo_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_type SRCS type_test.cc DEPS cinncore)
cinn_cc_test(test_topo_walker SRCS topo_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_shared SRCS shared_test.cc DEPS cinncore)
cinn_cc_test(test_is_reachable_predicator SRCS is_reachable_predicator_test.cc
             DEPS gtest glog)
cinn_cc_test(test_integer_set SRCS integer_set_test.cc DEPS cinncore)
if(WITH_CUDA)
  cinn_nv_test(test_fp16_bf16_cuda SRCS float16_bfloat16_cuda_test.cu DEPS
               gtest glog)
endif()
cinn_cc_test(test_fp16_bf16_host SRCS float16_bfloat16_host_test.cc DEPS gtest
             glog)
