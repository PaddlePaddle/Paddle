if(WITH_NGRAPH)
  cc_library(ngraph_engine SRCS ngraph_engine.cc DEPS ngraph_bridge framework_proto)
  op_library(ngraph_engine_op DEPS ngraph_engine op_registry op_info device_context)
endif()
