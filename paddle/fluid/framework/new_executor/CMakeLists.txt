add_subdirectory(garbage_collector)
add_subdirectory(instruction)
add_subdirectory(interpreter)
add_subdirectory(workqueue)
add_subdirectory(pir_adaptor)

set(STANDALONE_EXECUTOR_SRCS
    feed_fetch_utils.cc interpretercore.cc new_executor_defs.cc
    standalone_executor.cc program_interpreter.cc pir_interpreter.cc)

set(STANDALONE_EXECUTOR_DEPS
    interpreter
    interpretercore_garbage_collector
    workqueue
    op_dialect_vjp
    pir_transforms
    pir_adaptor
    program_translator
    instruction_base
    pir
    plan)

cc_library(
  standalone_executor
  SRCS ${STANDALONE_EXECUTOR_SRCS}
  DEPS ${STANDALONE_EXECUTOR_DEPS})

cc_library(
  staticgraph_executor_statistics
  SRCS executor_statistics.cc
  DEPS enforce glog phi common)

# file(GLOB_RECURSE new_executor_srcs "*.cc")

# if(NOT(WITH_CINN AND NOT CINN_ONLY))
#     list(
#         REMOVE_ITEM
#         new_executor_srcs
#         ${CMAKE_CURRENT_SOURCE_DIR}/instruction/cinn_jit_instruction.cc)
# endif()

# set(new_executor_deps pir program_translator op_dialect_vjp pir_transforms enforce glog phi common garbage_collector executor_gc_helper framework_proto)

# set(INTERPRETER_DEPS
#     buffered_reader
#     device_context
#     global_utils
#     op_registry
#     scope
#     data_feed_proto
#     ops_extra_info
#     heter_service_proto
#     trainer_desc_proto
#     lod_rank_table
#     framework_io
#     string_helper
#     fleet_wrapper
#     heter_wrapper
#     ps_gpu_wrapper
#     box_wrapper
#     lodtensor_printer
#     feed_fetch_method
#     graph_to_program_pass
#     variable_helper
#     timer
#     monitor
#     nan_inf_utils
#     scope
#     ${DEVICE_EVENT_LIBS}
#     glog)

# if(WITH_CINN AND NOT CINN_ONLY)
#   set(new_executor_deps ${new_executor_deps} ${INTERPRETER_DEPS})
# endif()

# cc_library( SRCS ${new_executor_srcs} DEPS ${new_executor_deps})
