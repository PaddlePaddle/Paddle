cc_library(
  interpretercore_garbage_collector
<<<<<<< HEAD
  SRCS garbage_collector.cc event_garbage_collector.cc fast_garbage_collector.cc
       no_event_garbage_collector.cc
  DEPS garbage_collector executor_gc_helper)
=======
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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
