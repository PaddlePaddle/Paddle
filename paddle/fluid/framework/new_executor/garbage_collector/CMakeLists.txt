cc_library(
  interpretercore_garbage_collector NOT_FOR_INFER
  SRCS garbage_collector.cc event_garbage_collector.cc fast_garbage_collector.cc
       no_event_garbage_collector.cc
  DEPS garbage_collector executor_gc_helper)
