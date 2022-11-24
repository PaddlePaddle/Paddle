cc_library(
  interpretercore_garbage_collector
<<<<<<< HEAD
  SRCS garbage_collector.cc
  DEPS garbage_collector)
cc_library(
  interpretercore_event_garbage_collector
  SRCS event_garbage_collector.cc
  DEPS interpretercore_garbage_collector)

if(WITH_GPU OR WITH_ROCM)
  if(WITH_GPU)
    nv_library(
      interpretercore_fast_garbage_collector
      SRCS fast_garbage_collector.cc
      DEPS interpretercore_garbage_collector)
  elseif(WITH_ROCM)
    hip_library(
      interpretercore_fast_garbage_collector
      SRCS fast_garbage_collector.cc
      DEPS interpretercore_garbage_collector)
  endif()
endif()
=======
  SRCS garbage_collector.cc event_garbage_collector.cc fast_garbage_collector.cc
       no_event_garbage_collector.cc
  DEPS garbage_collector executor_gc_helper)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
