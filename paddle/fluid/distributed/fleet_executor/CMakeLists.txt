proto_library(fleet_executor_desc_proto SRCS fleet_executor_desc.proto)
cc_library(fleet_executor SRCS fleet_executor.cc DEPS fleet_executor_desc_proto)

if(WITH_PYTHON)
  py_proto_compile(fleet_executor_desc_py_proto SRCS fleet_executor_desc.proto)
endif()
