<<<<<<< HEAD
cc_library(
  dependency_builder
  SRCS dependency_builder.cc
  DEPS operator)
=======
set(INTERPRETER_SRCS data_transfer.cc dependency_builder.cc event_manager.cc
                     execution_config.cc interpreter_util.cc)

set(INTERPRETER_DEPS
    device_context
    op_registry
    scope
    framework_proto
    data_feed_proto
    ops_extra_info
    heter_service_proto
    trainer_desc_proto
    glog
    lod_rank_table
    fs
    shell
    fleet_wrapper
    heter_wrapper
    ps_gpu_wrapper
    box_wrapper
    lodtensor_printer
    feed_fetch_method
    graph_to_program_pass
    variable_helper
    timer
    monitor
    nan_inf_utils
    enforce
    scope
    glog
    ${DEVICE_EVENT_LIBS}
    glog)

cc_library(
  interpreter
  SRCS ${INTERPRETER_SRCS}
  DEPS ${INTERPRETER_DEPS})
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
