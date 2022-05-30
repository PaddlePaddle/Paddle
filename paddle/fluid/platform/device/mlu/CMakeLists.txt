
if(NOT WITH_MLU)
    return()
endif()

cc_test(mlu_enforce_test SRCS enforce_test.cc DEPS stringpiece)
cc_library(mlu_info SRCS mlu_info.cc DEPS enforce glog monitor neuware_lib)
cc_library(mlu_stream SRCS mlu_stream.cc DEPS boost mlu_info stream_callback_manager eigen3 ${MKLDNN_CTX_DEPS})
cc_library(mlu_device_context SRCS device_context.cc DEPS mlu_stream)
cc_test(mlu_device_context_test SRCS device_context_test.cc DEPS mlu_device_context)
cc_library(mlu_collective_helper SRCS mlu_collective_helper.cc DEPS mlu_stream mlu_info)
cc_library(mlu_resource_pool SRCS mlu_resource_pool.cc DEPS mlu_info)
