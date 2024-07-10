if(WITH_PSCORE)
  set(DISTRIBUTE_COMPILE_FLAGS
      "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor"
  )
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
  endif()
  set_source_files_properties(
    reduce_op_handle.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()

set(op_handle_srcs
    nan_inf_utils_detail.cc
    all_reduce_op_handle.cc
    fused_all_reduce_op_handle.cc
    reduce_op_handle.cc
    broadcast_op_handle.cc
    fused_broadcast_op_handle.cc
    var_handle.cc
    op_handle_base.cc
    scale_loss_grad_op_handle.cc
    share_tensor_buffer_functor.cc
    computation_op_handle.cc
    share_tensor_buffer_op_handle.cc
    variable_visitor.cc
    eager_deletion_op_handle.cc)

if(WITH_DGC)
  set(op_handle_srcs ${op_handle_srcs} sparse_all_reduce_op_handle.cc)
endif()

set(op_handle_deps
    pass
    operator
    place
    framework_proto
    node
    device_context
    op_registry
    lod_tensor
    selected_rows_utils
    reference_count_pass_helper)

if(WITH_ONEDNN)
  set(op_handle_deps ${op_handle_deps} onednn)
endif()

if(WITH_DGC)
  set(op_handle_deps ${op_handle_deps} dgc)
endif()

if(WITH_GPU)
  nv_library(
    detail_op_handle
    SRCS ${op_handle_srcs}
    DEPS ${op_handle_deps})
elseif(WITH_ROCM)
  hip_library(
    detail_op_handle
    SRCS ${op_handle_srcs}
    DEPS ${op_handle_deps})
else()
  cc_library(
    detail_op_handle
    SRCS ${op_handle_srcs}
    DEPS ${op_handle_deps})
endif()

add_dependencies(detail_op_handle framework_proto auto_parallel_proto xxhash)

set(IR_PASS_DEPS
    graph_viz_pass
    multi_devices_graph_pass
    fuse_elewise_add_act_pass
    fuse_bn_act_pass
    fuse_bn_add_act_pass
    multi_batch_merge_pass
    fuse_relu_depthwise_conv_pass
    lock_free_optimize_pass
    add_reader_dependency_pass
    modify_op_lock_and_record_event_pass
    coalesce_grad_tensor_pass
    fuse_adam_op_pass
    fuse_sgd_op_pass
    fuse_momentum_op_pass
    sync_batch_norm_pass
    runtime_context_cache_pass
    graph_to_program_pass
    fuse_gemm_epilogue_pass
    fused_attention_pass
    fuse_adamw_op_pass
    fused_feedforward_pass
    delete_dropout_op_pass)

if(WITH_CUDNN_FRONTEND)
  set(IR_PASS_DEPS ${IR_PASS_DEPS} fuse_dot_product_attention_pass)
  set(IR_PASS_DEPS ${IR_PASS_DEPS} fuse_resunit_pass)
endif()

if(NOT APPLE
   AND NOT WIN32
   AND (WITH_GPU OR WITH_ROCM))
  set(IR_PASS_DEPS ${IR_PASS_DEPS} fusion_group_pass)
endif()
cc_library(
  build_strategy
  SRCS build_strategy.cc
  DEPS pass_builder ${IR_PASS_DEPS})

if(WITH_ONEDNN)
  target_link_libraries(build_strategy onednn_placement_pass)
endif()
