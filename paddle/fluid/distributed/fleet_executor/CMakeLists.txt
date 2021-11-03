if(WITH_DISTRIBUTE)
  proto_library(fleet_executor_desc_proto SRCS fleet_executor_desc.proto)
  proto_library(interceptor_message_proto SRCS interceptor_message.proto)
  cc_library(fleet_executor SRCS fleet_executor.cc carrier.cc
          interceptor.cc interceptor_message_service.cc message_bus.cc
          DEPS fleet_executor_desc_proto interceptor_message_proto)
  set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
  set_source_files_properties(message_bus.h PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  set_source_files_properties(message_bus.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  set_source_files_properties(carrier.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  set_source_files_properties(interceptor_message_service.h PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  set_source_files_properties(interceptor_message_service.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

  if(WITH_PYTHON)
    py_proto_compile(fleet_executor_desc_py_proto SRCS fleet_executor_desc.proto)
  endif()
endif()
