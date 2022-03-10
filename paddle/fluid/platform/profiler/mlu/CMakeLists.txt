if(NOT WITH_MLU)
    return()
endif()

cc_library(mlu_tracer SRCS mlu_tracer.cc cnpapi_data_process.cc DEPS workqueue_utils enforce glog mlu_info)
