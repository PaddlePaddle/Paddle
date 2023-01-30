cc_library(
  interpretercore_garbage_collector
  SRCS garbage_collector.cc event_garbage_collector.cc fast_garbage_collector.cc
       no_event_garbage_collector.cc
<<<<<<< HEAD
  DEPS garbage_collector executor_gc_helper)
=======
  DEPS garbage_collector)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
