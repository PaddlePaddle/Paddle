if(WITH_CUSTOM_DEVICE)
  cc_library(
    custom_device_resource_pool
    SRCS custom_device_resource_pool.cc
    DEPS phi glog enforce monitor)
  cc_test(
    custom_device_test
    SRCS custom_device_test.cc
    DEPS phi gradient_accumulator)
endif()
