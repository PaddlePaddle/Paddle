proto_library(fleet_executor_desc_proto SRCS fleet_executor_desc.proto)
proto_library(interceptor_message_proto SRCS interceptor_message.proto)
cc_library(fleet_executor SRCS fleet_executor.cc carrier.cc interceptor.cc interceptor_message_service.cc message_bus.cc DEPS fleet_executor_desc_proto interceptor_message_proto)

if(WITH_PYTHON)
  py_proto_compile(fleet_executor_desc_py_proto SRCS fleet_executor_desc.proto)
endif()
