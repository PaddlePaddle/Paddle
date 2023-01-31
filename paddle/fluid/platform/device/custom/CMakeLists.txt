if(WITH_CUSTOM_DEVICE)
  cc_library(
    custom_device_resource_pool
    SRCS custom_device_resource_pool.cc
    DEPS gflags glog enforce monitor)
endif()
