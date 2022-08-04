cc_library(
  interpretercore_garbage_collector
  SRCS garbage_collector.cc event_garbage_collector.cc fast_garbage_collector.cc
       no_event_garbage_collector.cc
  DEPS garbage_collector)
