set(INTERPRETERCORE_DEPS op_registry device_context scope framework_proto data_feed_proto heter_service_proto trainer_desc_proto glog 
lod_rank_table fs shell fleet_wrapper heter_wrapper ps_gpu_wrapper box_wrapper lodtensor_printer feed_fetch_method
graph_to_program_pass variable_helper timer monitor)

cc_library(workqueue SRCS workqueue.cc DEPS enforce)
cc_library(interpretercore_garbage_collector SRCS interpretercore_garbage_collector.cc DEPS workqueue ${DEVICE_EVENT_LIBS})
cc_library(interpretercore_util SRCS interpretercore_util.cc DEPS ${INTERPRETERCORE_DEPS})
cc_library(event_manager SRCS event_manager.cc DEPS ${DEVICE_EVENT_LIBS} glog)
cc_library(stream_analyzer SRCS stream_analyzer.cc DEPS ${DEVICE_EVENT_LIBS} glog device_context)
cc_library(interpretercore SRCS interpretercore.cc DEPS workqueue ${DEVICE_EVENT_LIBS} interpretercore_util interpretercore_garbage_collector stream_analyzer event_manager)
cc_library(standalone_executor SRCS standalone_executor.cc DEPS interpretercore)
cc_test(workqueue_test SRCS workqueue_test.cc DEPS workqueue)
# cc_binary(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} profiler)
