cc_library(host_event_recorder SRCS host_event_recorder.cc DEPS os_info)
cc_library(chrometracinglogger SRCS chrometracing_logger.cc)
cc_library(event_node SRCS event_node.cc)
cc_test(test_event_node SRCS test_event_node.cc DEPS event_node chrometracinglogger)
