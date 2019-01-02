if(WITH_NGRAPH)
  cc_library(ngraph_bridge SRCS ngraph_bridge.cc DEPS operator framework_proto ngraph)
  cc_library(ngraph_operator SRCS ngraph_operator.cc DEPS ngraph_bridge operator op_info device_context tensor scope glog
             shape_inference data_transform lod_tensor profiler)
endif(WITH_NGRAPH)
