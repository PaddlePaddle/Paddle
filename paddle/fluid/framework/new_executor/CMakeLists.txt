cc_library(workqueue SRCS workqueue.cc)
cc_library(interpretercore SRCS interpretercore.cc DEPS op_registry
            device_context scope framework_proto data_feed_proto heter_service_proto trainer_desc_proto glog
            lod_rank_table fs shell fleet_wrapper heter_wrapper ps_gpu_wrapper box_wrapper lodtensor_printer feed_fetch_method
            graph_to_program_pass variable_helper timer monitor)
cc_library(standalone_executor SRCS standalone_executor.cc DEPS interpretercore)
cc_test(workqueue_test SRCS workqueue_test.cc DEPS workqueue)
# cc_binary(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} profiler)
