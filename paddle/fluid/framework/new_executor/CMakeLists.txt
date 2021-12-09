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
# cc_binary(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} profiler)
