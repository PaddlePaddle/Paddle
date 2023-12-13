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
