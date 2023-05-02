if(WITH_CUSTOM_DEVICE)
  cc_library(
    custom_device_resource_pool
    SRCS custom_device_resource_pool.cc
    DEPS gflags glog enforce monitor)
  cc_test(
    custom_device_test
    SRCS custom_device_test.cc
    DEPS phi_tensor_utils phi_backends phi_device_context gradient_accumulator)
endif()
