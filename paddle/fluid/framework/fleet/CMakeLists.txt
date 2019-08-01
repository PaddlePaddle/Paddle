if(WITH_PSLIB)
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope pslib_brpc pslib)
else()
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope)
endif(WITH_PSLIB)

cc_library(nccl_wrapper SRCS nccl_wrapper.cc DEPS framework_proto variable_helper scope)
cc_library(boxps SRCS boxps.cc)
cc_library(box_wrapper SRCS box_wrapper.cc DEPS boxps)
