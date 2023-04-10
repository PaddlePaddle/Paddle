add_subdirectory(garbage_collector)
add_subdirectory(interpreter)
add_subdirectory(workqueue)

set(STANDALONE_EXECUTOR_SRCS interpretercore.cc new_executor_defs.cc
                             standalone_executor.cc)

set(STANDALONE_EXECUTOR_DEPS interpreter interpretercore_garbage_collector
                             workqueue)

cc_library(
  standalone_executor
  SRCS ${STANDALONE_EXECUTOR_SRCS}
  DEPS ${STANDALONE_EXECUTOR_DEPS})

cc_library(
  staticgraph_executor_statistics
  SRCS executor_statistics.cc
  DEPS enforce glog phi_os_info)
