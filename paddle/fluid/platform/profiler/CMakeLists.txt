cc_library(host_event_recorder SRCS host_event_recorder.cc DEPS os_info)
cc_library(host_tracer SRCS host_tracer.cc DEPS host_event_recorder)
cc_library(cuda_tracer SRCS cuda_tracer.cc DEPS workqueue_utils enforce glog)
cc_library(new_profiler SRCS profiler.cc DEPS host_tracer cuda_tracer)
