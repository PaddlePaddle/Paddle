add_subdirectory(lib)
cc_library(
  phi_capi
  SRCS all.cc
  DEPS phi_c_data_type
       phi_c_device_context
       phi_c_int_array
       phi_c_kernel_context
       phi_c_kernel_factory
       phi_c_kernel_registry
       phi_c_place
       phi_c_scalar
       phi_c_tensor)
