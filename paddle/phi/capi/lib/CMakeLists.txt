cc_library(
  phi_c_data_type
  SRCS c_data_type.cc
  DEPS dense_tensor)

cc_library(
  phi_c_device_context
  SRCS c_device_context.cc
  DEPS phi_backends)

cc_library(
  phi_c_int_array
  SRCS c_int_array.cc
  DEPS int_array)

cc_library(
  phi_c_kernel_context
  SRCS c_kernel_context.cc
  DEPS kernel_context)

cc_library(
  phi_c_kernel_factory
  SRCS c_kernel_factory.cc
  DEPS kernel_factory)

cc_library(
  phi_c_kernel_registry
  SRCS c_kernel_registry.cc
  DEPS dense_tensor)

cc_library(
  phi_c_place
  SRCS c_place.cc
  DEPS phi_place)

cc_library(
  phi_c_scalar
  SRCS c_scalar.cc
  DEPS scalar)

cc_library(
  phi_c_tensor
  SRCS c_tensor.cc
  DEPS dense_tensor)
