cc_library(var_handle SRCS var_handle.cc DEPS place)
cc_library(op_handle_base SRCS op_handle_base.cc DEPS var_handle device_context)
cc_library(scale_loss_grad_op_handle SRCS scale_loss_grad_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(fetch_op_handle SRCS fetch_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
nv_library(nccl_all_reduce_op_handle SRCS nccl_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
        dynload_cuda)
cc_library(computation_op_handle SRCS computation_op_handle.cc DEPS framework_proto scope place operator op_registry)

cc_library(ssa_graph SRCS ssa_graph.cc DEPS var_handle op_handle_base)
cc_library(ssa_graph_builder SRCS ssa_graph_builder.cc DEPS ssa_graph)

if(WITH_GPU)
    set(multi_devices_graph_builder_deps nccl_all_reduce_op_handle)
else()
    set(multi_devices_graph_builder_deps)
endif()
cc_library(multi_devices_graph_builder SRCS multi_devices_graph_builder.cc DEPS ssa_graph_builder computation_op_handle
            scale_loss_grad_op_handle ${multi_devices_graph_builder_deps})
cc_library(ssa_graph_executor SRCS ssa_graph_executor.cc DEPS ssa_graph)
cc_library(threaded_ssa_graph_executor SRCS threaded_ssa_graph_executor.cc DEPS fetch_op_handle ssa_graph_executor scope
        simple_threadpool device_context)
