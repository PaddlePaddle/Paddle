cc_library(var_handle SRCS var_handle.cc DEPS place framework_proto node)
cc_library(op_handle_base SRCS op_handle_base.cc DEPS var_handle device_context lod_tensor)
cc_library(scale_loss_grad_op_handle SRCS scale_loss_grad_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(fetch_op_handle SRCS fetch_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(computation_op_handle SRCS computation_op_handle.cc DEPS framework_proto scope place operator op_registry)
cc_library(rpc_op_handle SRCS rpc_op_handle.cc DEPS framework_proto scope place operator op_registry)

cc_library(multi_devices_helper SRCS multi_devices_helper.cc DEPS graph graph_helper)
cc_library(multi_devices_graph_print_pass SRCS multi_devices_graph_print_pass.cc DEPS multi_devices_helper)
cc_library(multi_devices_graph_check_pass SRCS multi_devices_graph_check_pass.cc DEPS multi_devices_helper)

cc_library(variable_visitor SRCS variable_visitor.cc DEPS lod_tensor selected_rows)

if(WITH_GPU)
    nv_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor)
    nv_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope ddim dynload_cuda)
    nv_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor dynload_cuda)
    nv_library(fused_broadcast_op_handle SRCS fused_broadcast_op_handle.cc DEPS broadcast_op_handle)

else()
    cc_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
             variable_visitor)
    cc_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope ddim)
    cc_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)
    cc_library(fused_broadcast_op_handle SRCS fused_broadcast_op_handle.cc DEPS broadcast_op_handle)
endif()

cc_library(data_balance_op_handle SRCS data_balance_op_handle.cc DEPS op_handle_base scope lod_tensor)
cc_library(gather_op_handle SRCS gather_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)
cc_library(fuse_vars_op_handle SRCS fuse_vars_op_handle.cc DEPS op_handle_base scope)

if(WITH_GPU)
  cc_library(reference_count_pass SRCS reference_count_pass.cc DEPS computation_op_handle scale_loss_grad_op_handle rpc_op_handle
          all_reduce_op_handle reduce_op_handle broadcast_op_handle data_balance_op_handle graph graph_helper pass)
endif()

cc_library(multi_devices_graph_pass SRCS multi_devices_graph_pass.cc DEPS multi_devices_helper computation_op_handle
        scale_loss_grad_op_handle rpc_op_handle all_reduce_op_handle reduce_op_handle broadcast_op_handle data_balance_op_handle fused_broadcast_op_handle)

if(WITH_GPU)
  cc_library(ssa_graph_executor SRCS ssa_graph_executor.cc DEPS graph framework_proto reference_count_pass)
else()
  cc_library(ssa_graph_executor SRCS ssa_graph_executor.cc DEPS graph framework_proto)
endif()

cc_library(threaded_ssa_graph_executor SRCS threaded_ssa_graph_executor.cc DEPS fetch_op_handle ssa_graph_executor scope
        simple_threadpool device_context)

cc_test(broadcast_op_test SRCS broadcast_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context broadcast_op_handle)
cc_test(gather_op_test SRCS gather_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context gather_op_handle)
cc_library(scope_buffered_ssa_graph_executor SRCS scope_buffered_ssa_graph_executor.cc DEPS ssa_graph_executor)
#cc_test(reduce_op_handle_test SRCS reduce_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
#        device_context reduce_op_handle )
cc_library(fast_threaded_ssa_graph_executor SRCS fast_threaded_ssa_graph_executor.cc
        DEPS fetch_op_handle ssa_graph_executor scope simple_threadpool device_context)

cc_library(build_strategy SRCS build_strategy.cc DEPS
        graph_viz_pass multi_devices_graph_pass
        multi_devices_graph_print_pass multi_devices_graph_check_pass
        fuse_elewise_add_act_pass multi_batch_merge_pass)
