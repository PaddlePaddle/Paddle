set(INTERPRETERCORE_DEPS op_registry device_context scope framework_proto data_feed_proto heter_service_proto trainer_desc_proto glog 
lod_rank_table fs shell fleet_wrapper heter_wrapper ps_gpu_wrapper box_wrapper lodtensor_printer feed_fetch_method
graph_to_program_pass variable_helper timer monitor nan_inf_utils interpretercore_event_garbage_collector)

if(WITH_GPU)
list(APPEND INTERPRETERCORE_DEPS interpretercore_fast_garbage_collector)
endif()

add_subdirectory(workqueue)

cc_library(data_transfer SRCS data_transfer.cc DEPS enforce scope glog)
cc_library(new_executor_defs SRCS new_executor_defs.cc DEPS enforce glog scope)
cc_library(interpretercore_garbage_collector SRCS interpretercore_garbage_collector.cc DEPS garbage_collector)
cc_library(interpretercore_event_garbage_collector SRCS interpretercore_event_garbage_collector.cc DEPS interpretercore_garbage_collector)
cc_library(interpretercore_util SRCS interpretercore_util.cc DEPS ${INTERPRETERCORE_DEPS} workqueue new_executor_defs data_transfer)
cc_library(event_manager SRCS event_manager.cc DEPS ${DEVICE_EVENT_LIBS} glog new_executor_defs)
cc_library(stream_analyzer SRCS stream_analyzer.cc DEPS ${DEVICE_EVENT_LIBS} glog device_context new_executor_defs)
cc_library(interpretercore SRCS interpretercore.cc DEPS workqueue ${DEVICE_EVENT_LIBS} interpretercore_util interpretercore_event_garbage_collector stream_analyzer event_manager)
cc_library(standalone_executor SRCS standalone_executor.cc DEPS interpretercore)

if(WITH_GPU OR WITH_ROCM)
    if(WITH_GPU)
        nv_library(interpretercore_fast_garbage_collector SRCS interpretercore_fast_garbage_collector.cc DEPS interpretercore_garbage_collector)
    elseif(WITH_ROCM)
        hip_library(interpretercore_fast_garbage_collector SRCS interpretercore_fast_garbage_collector.cc DEPS interpretercore_garbage_collector)
    endif()
    
    target_link_libraries(interpretercore interpretercore_fast_garbage_collector)
endif()

# cc_binary(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} profiler)
# skip win32 since wget is not installed by default on windows machine.
# skip COVERAGE_CI since the test runs slowly because of instrumentation.
if (WITH_TESTING AND NOT WIN32 AND NOT WITH_COVERAGE AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
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
    
    # All deps of the operators above, part of GLOB_OPERATOR_DEPS.
    set(OP_DEPS 
        generator
        softmax
        selected_rows_functor
        jit_kernel_helper
        concat_and_split
        cross_entropy)

    cc_test(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${OPS} ${OP_DEPS})
    set_tests_properties(standalone_executor_test PROPERTIES TIMEOUT 100)

    add_dependencies(standalone_executor_test download_program)
    if (WITH_PROFILER)
        target_link_libraries(standalone_executor_test profiler)
        add_dependencies(standalone_executor_test profiler)
    endif()
endif()
