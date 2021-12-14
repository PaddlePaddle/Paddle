set(INTERPRETERCORE_DEPS op_registry device_context scope framework_proto data_feed_proto heter_service_proto trainer_desc_proto glog 
lod_rank_table fs shell fleet_wrapper heter_wrapper ps_gpu_wrapper box_wrapper lodtensor_printer feed_fetch_method
graph_to_program_pass variable_helper timer monitor nan_inf_utils)

cc_library(data_transfer SRCS data_transfer.cc DEPS enforce scope glog)
cc_library(workqueue SRCS workqueue.cc workqueue_utils.cc DEPS enforce)
cc_library(new_executor_defs SRCS new_executor_defs.cc DEPS enforce glog scope)
cc_library(interpretercore_garbage_collector SRCS interpretercore_garbage_collector.cc DEPS workqueue ${DEVICE_EVENT_LIBS} executor_gc_helper)
cc_library(interpretercore_util SRCS interpretercore_util.cc DEPS ${INTERPRETERCORE_DEPS} workqueue new_executor_defs data_transfer)
cc_library(event_manager SRCS event_manager.cc DEPS ${DEVICE_EVENT_LIBS} glog new_executor_defs)
cc_library(stream_analyzer SRCS stream_analyzer.cc DEPS ${DEVICE_EVENT_LIBS} glog device_context new_executor_defs)
cc_library(interpretercore SRCS interpretercore.cc DEPS workqueue ${DEVICE_EVENT_LIBS} interpretercore_util interpretercore_garbage_collector stream_analyzer event_manager)
cc_library(standalone_executor SRCS standalone_executor.cc DEPS interpretercore)
cc_test(workqueue_test SRCS workqueue_test.cc DEPS workqueue)

if (WITH_TESTING AND NOT WIN32)
    add_custom_target(
        download_program
        COMMAND wget -nc https://paddle-ci.gz.bcebos.com/new_exec/lm_main_program 
        COMMAND wget -nc https://paddle-ci.gz.bcebos.com/new_exec/lm_startup_program 
    )
    
    # all operators used in the program
    set(OPS  
        fill_constant_op
        uniform_random_op
        lookup_table_op
        transpose_op
        reshape_op
        split_op
        slice_op
        concat_op
        matmul_op
        elementwise_add_op
        elementwise_mul_op
        softmax_with_cross_entropy_op
        reduce_mean_op
        reduce_sum_op
        activation_op
        sum_op
        elementwise_max_op
        elementwise_div_op
        sgd_op
        squared_l2_norm_op
        memcpy_h2d_op
        memcpy_d2h_op)
    
    # All deps of operator, same as GLOB_OPERATOR_DEPS.
    # WHY list here manually?
    # Directory framework is processed before operators in fluid/CMakeLists.txt,
    # So the variable GLOB_OPERATOR_DEPS is not defined at here.
    set(OP_DEPS 
        nccl_common
        xxhash
        executor
        pocketfft
        pten
        pten_api_utils
        selected_rows_functor
        selected_rows
        lapack_function
        lod_tensor
        maxouting
        unpooling
        pooling
        lod_rank_table
        context_project
        sequence_pooling
        segment_pooling
        executor
        device_memory_aligment
        generator
        dynload_warpctc
        sequence_padding
        sequence_scale
        cos_sim_functor
        memory
        jit_kernel_helper
        concat_and_split
        cross_entropy
        softmax
        vol2col
        im2col
        sampler
        sample_prob
        tree2col
        sequence2batch
        lstm_compute
        matrix_bit_code
        gru_compute
        activation_functions
        beam_search
        fc
        matrix_inverse
        matrix_solve
        box_wrapper
        boost
        ps_gpu_wrapper
        common_infer_shape_functions
        eigen_function
        depthwise_conv
        prelu
        bert_encoder_functor
        device_memory_aligment
        layer
        tensor_formatter
        op_version_registry)

    cc_test(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${OPS} ${OP_DEPS})
    set_tests_properties(standalone_executor_test PROPERTIES TIMEOUT 100)

    add_dependencies(standalone_executor_test download_program)
    if (WITH_PROFILER)
        target_link_libraries(standalone_executor_test profiler)
        add_dependencies(standalone_executor_test profiler)
    endif()
endif()
