cc_library(host_tracer SRCS host_tracer.cc DEPS enforce)
cc_library(cuda_tracer SRCS cuda_tracer.cc DEPS workqueue_utils enforce glog)
cc_library(new_profiler SRCS profiler.cc DEPS host_tracer cuda_tracer)
cc_test(new_profiler_test SRCS profiler_test.cc DEPS new_profiler)
