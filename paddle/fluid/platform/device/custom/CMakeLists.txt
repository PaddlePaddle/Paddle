if(WITH_CUSTOM_DEVICE)
  cc_library(
    custom_device_resource_pool
    SRCS custom_device_resource_pool.cc
    DEPS phi common glog monitor)
endif()
