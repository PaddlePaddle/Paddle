cc_library(gpu_event SRCS gpu_event.cc DEPS parallel_executor)
cc_library(paddle_stream SRCS paddle_stream.cc DEPS parallel_executor)
cc_library(gpu_event_impl SRCS gpu_event_impl.cc DEPS )
cc_library(gpu_stream SRCS gpu_stream.cc DEPS )
