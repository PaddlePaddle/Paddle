if(WITH_MLU)
  set(MLU_INFO mlu_info)
endif()

cc_library(mlu_tracer SRCS mlu_tracer.cc cnpapi_data_process.cc DEPS workqueue_utils enforce glog ${MLU_INFO})
