cc_library(host_event_recorder SRCS host_event_recorder.cc DEPS os_info)
cc_library(cuda_tracer SRCS cuda_tracer.cc DEPS workqueue_utils dynload_cuda)
