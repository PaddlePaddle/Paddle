cc_library(var_handle SRCS var_handle.cc DEPS place)
cc_library(op_handle_base SRCS op_handle_base.cc DEPS var_handle device_context lod_tensor)
cc_library(scale_loss_grad_op_handle SRCS scale_loss_grad_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(fetch_op_handle SRCS fetch_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(computation_op_handle SRCS computation_op_handle.cc DEPS framework_proto scope place operator op_registry)
cc_library(rpc_op_handle SRCS rpc_op_handle.cc DEPS framework_proto scope place operator op_registry)

cc_library(ssa_graph SRCS ssa_graph.cc DEPS var_handle op_handle_base)
cc_library(ssa_graph_builder SRCS ssa_graph_builder.cc DEPS ssa_graph)
cc_library(ssa_graph_printer SRCS ssa_graph_printer.cc DEPS ssa_graph_builder)
cc_library(ssa_graph_checker SRCS ssa_graph_checker.cc DEPS ssa_graph_builder)

cc_library(variable_visitor SRCS variable_visitor.cc DEPS lod_tensor selected_rows)

if(WITH_GPU)
    nv_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor)
    nv_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope ddim dynload_cuda)
    nv_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor dynload_cuda)

else()
    cc_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
             variable_visitor)
    cc_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope ddim)
    cc_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)
endif()

cc_library(gather_op_handle SRCS gather_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)
cc_library(fuse_vars_op_handle SRCS fuse_vars_op_handle.cc DEPS op_handle_base scope)

cc_library(multi_devices_graph_builder SRCS multi_devices_graph_builder.cc DEPS ssa_graph_builder computation_op_handle
        scale_loss_grad_op_handle rpc_op_handle all_reduce_op_handle reduce_op_handle broadcast_op_handle)


cc_library(ssa_graph_builder_factory SRCS ssa_graph_builder_factory.cc DEPS multi_devices_graph_builder ssa_graph_printer ssa_graph_checker)

cc_library(ssa_graph_executor SRCS ssa_graph_executor.cc DEPS ssa_graph framework_proto)
cc_library(threaded_ssa_graph_executor SRCS threaded_ssa_graph_executor.cc DEPS fetch_op_handle ssa_graph_executor scope
        simple_threadpool device_context)

cc_test(broadcast_op_test SRCS broadcast_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context broadcast_op_handle)
cc_test(gather_op_test SRCS gather_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context gather_op_handle)
cc_library(scope_buffered_ssa_graph_executor SRCS scope_buffered_ssa_graph_executor.cc DEPS ssa_graph_executor)
#cc_test(reduce_op_handle_test SRCS reduce_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
#        device_context reduce_op_handle )
