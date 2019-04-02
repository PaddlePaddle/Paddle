cc_library(var_handle SRCS var_handle.cc DEPS place framework_proto node)
cc_library(op_handle_base SRCS op_handle_base.cc DEPS var_handle device_context lod_tensor)
cc_library(op_graph_view SRCS op_graph_view.cc DEPS op_handle_base)
cc_library(scale_loss_grad_op_handle SRCS scale_loss_grad_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(fetch_op_handle SRCS fetch_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(computation_op_handle SRCS computation_op_handle.cc DEPS framework_proto scope place operator op_registry)
cc_library(rpc_op_handle SRCS rpc_op_handle.cc DEPS framework_proto scope place operator op_registry)
cc_library(fetch_barrier_op_handle SRCS fetch_barrier_op_handle.cc DEPS framework_proto scope place operator op_registry)

cc_library(multi_devices_helper SRCS multi_devices_helper.cc DEPS graph graph_helper)
cc_library(multi_devices_graph_print_pass SRCS multi_devices_graph_print_pass.cc DEPS multi_devices_helper)
cc_library(multi_devices_graph_check_pass SRCS multi_devices_graph_check_pass.cc DEPS multi_devices_helper)

cc_library(alloc_continuous_space_for_grad_pass SRCS alloc_continuous_space_for_grad_pass.cc DEPS graph graph_helper)
cc_library(fuse_adam_op_pass SRCS fuse_adam_op_pass.cc fuse_optimizer_op_pass.cc DEPS graph graph_helper)
cc_library(fuse_sgd_op_pass SRCS fuse_sgd_op_pass.cc fuse_optimizer_op_pass.cc DEPS graph graph_helper)

cc_library(variable_visitor SRCS variable_visitor.cc DEPS lod_tensor selected_rows)

if(WITH_DISTRIBUTE)
    if(NOT WITH_GRPC)
        set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
        set_source_files_properties(reduce_op_handle.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
    endif()
endif()

if(WITH_GPU)
    set(dgc_deps "")
    if(NOT WIN32 AND NOT ON_INFER)
        set(dgc_deps dgc)
    endif()
    nv_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor ${dgc_deps})
    nv_library(fused_all_reduce_op_handle SRCS fused_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor)
    if(WITH_DISTRIBUTE)
        nv_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
            ddim dynload_cuda selected_rows_functor sendrecvop_rpc)
    else()
        nv_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
            ddim dynload_cuda selected_rows_functor)
    endif()
    nv_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor dynload_cuda)
    nv_library(fused_broadcast_op_handle SRCS fused_broadcast_op_handle.cc DEPS broadcast_op_handle)

else()
    cc_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
             variable_visitor)
    cc_library(fused_all_reduce_op_handle SRCS fused_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            variable_visitor)
    if(WITH_DISTRIBUTE)
        cc_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
            ddim selected_rows_functor sendrecvop_rpc)
    else()
        cc_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
            ddim selected_rows_functor)
    endif()
    cc_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)
    cc_library(fused_broadcast_op_handle SRCS fused_broadcast_op_handle.cc DEPS broadcast_op_handle)
endif()

cc_library(gather_op_handle SRCS gather_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)

if(WITH_GPU)
cc_library(memory_optimize_helper SRCS memory_optimize_helper.cc DEPS graph graph_helper gpu_info)
else()
cc_library(memory_optimize_helper SRCS memory_optimize_helper.cc DEPS graph graph_helper cpu_info)
endif()

cc_library(memory_optimize_pass SRCS memory_optimize_pass.cc DEPS memory_optimize_helper pass)
cc_library(inplace_op_pass SRCS inplace_op_pass.cc DEPS memory_optimize_pass op_info)
cc_library(modify_op_lock_and_record_event_pass SRCS modify_op_lock_and_record_event_pass.cc DEPS computation_op_handle op_graph_view multi_devices_helper)
cc_library(reference_count_pass_helper SRCS reference_count_pass_helper.cc DEPS garbage_collector computation_op_handle)
cc_library(eager_deletion_op_handle SRCS eager_deletion_op_handle.cc DEPS lod_tensor selected_rows reference_count_pass_helper)
cc_library(while_op_eager_deletion_pass SRCS while_op_eager_deletion_pass.cc DEPS while_op_helper graph_helper pass computation_op_handle)
cc_library(eager_deletion_pass SRCS eager_deletion_pass.cc DEPS computation_op_handle eager_deletion_op_handle graph graph_helper pass while_op_eager_deletion_pass)
cc_library(reference_count_pass SRCS reference_count_pass.cc DEPS computation_op_handle graph graph_helper pass op_graph_view reference_count_pass_helper)

cc_library(sequential_execution_pass SRCS sequential_execution_pass.cc DEPS graph graph_helper pass)
cc_library(all_reduce_deps_pass SRCS all_reduce_deps_pass.cc DEPS graph graph_helper pass)

cc_library(multi_devices_graph_pass SRCS multi_devices_graph_pass.cc DEPS multi_devices_helper computation_op_handle
        scale_loss_grad_op_handle rpc_op_handle fetch_barrier_op_handle all_reduce_op_handle reduce_op_handle broadcast_op_handle fused_broadcast_op_handle)

cc_library(fuse_all_reduce_op_pass SRCS fuse_all_reduce_op_pass.cc DEPS graph graph_helper fused_all_reduce_op_handle)

set(SSA_GRAPH_EXECUTOR_DEPS graph framework_proto sequential_execution_pass modify_op_lock_and_record_event_pass all_reduce_deps_pass reference_count_pass eager_deletion_pass memory_optimize_pass inplace_op_pass)
if (WITH_GPU)
  list(APPEND SSA_GRAPH_EXECUTOR_DEPS reference_count_pass)
endif()
cc_test(memory_optimize_helper_test SRCS memory_optimize_helper_test.cc memory_optimize_helper.cc DEPS framework_proto graph graph_helper op_registry)
cc_library(ssa_graph_executor SRCS ssa_graph_executor.cc DEPS ${SSA_GRAPH_EXECUTOR_DEPS})

cc_library(threaded_ssa_graph_executor SRCS threaded_ssa_graph_executor.cc DEPS fetch_op_handle ssa_graph_executor scope
        simple_threadpool device_context)

cc_library(parallel_ssa_graph_executor SRCS parallel_ssa_graph_executor.cc DEPS threaded_ssa_graph_executor)

set(ASYNC_SSA_GRAPH_EXECUTOR_DEPS threaded_ssa_graph_executor)
if(WITH_DISTRIBUTE)
    list(APPEND ASYNC_SSA_GRAPH_EXECUTOR_DEPS communicator)
endif()
cc_library(async_ssa_graph_executor SRCS async_ssa_graph_executor.cc DEPS ${ASYNC_SSA_GRAPH_EXECUTOR_DEPS})

cc_test(broadcast_op_test SRCS broadcast_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context broadcast_op_handle)
cc_test(gather_op_test SRCS gather_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context gather_op_handle)
cc_library(scope_buffered_ssa_graph_executor SRCS scope_buffered_ssa_graph_executor.cc DEPS ssa_graph_executor)
#cc_test(reduce_op_handle_test SRCS reduce_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
#        device_context reduce_op_handle )
cc_library(fast_threaded_ssa_graph_executor SRCS fast_threaded_ssa_graph_executor.cc
        DEPS fetch_op_handle ssa_graph_executor scope simple_threadpool device_context)
cc_test(fused_broadcast_op_test SRCS fused_broadcast_op_handle_test.cc DEPS fused_broadcast_op_handle)

cc_library(build_strategy SRCS build_strategy.cc DEPS
        graph_viz_pass multi_devices_graph_pass
        multi_devices_graph_print_pass multi_devices_graph_check_pass
        fuse_elewise_add_act_pass multi_batch_merge_pass 
        fuse_relu_depthwise_conv_pass
        memory_optimize_pass lock_free_optimize_pass
        alloc_continuous_space_for_grad_pass fuse_all_reduce_op_pass
        fuse_adam_op_pass fuse_sgd_op_pass)
