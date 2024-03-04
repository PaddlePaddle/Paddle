# compatible utils used for fluid op system
add_subdirectory(compat)
add_subdirectory(distributed)

if(WITH_GPU)
  proto_library(external_error_proto SRCS external_error.proto)
endif()

collect_srcs(
  core_srcs
  SRCS
  flags.cc
  enforce.cc
  storage_properties.cc
  os_info.cc
  kernel_context.cc
  tensor_base.cc
  allocator.cc
  tensor_meta.cc
  lod_utils.cc
  threadpool.cc
  dense_tensor.cc
  dense_tensor_impl.cc
  sparse_coo_tensor.cc
  sparse_csr_tensor.cc
  string_tensor.cc
  tensor_array.cc
  extended_tensor.cc
  meta_tensor.cc
  infermeta_utils.cc
  selected_rows_impl.cc
  selected_rows.cc
  device_context.cc
  custom_kernel.cc
  mixed_vector.cc
  generator.cc
  kernel_factory.cc
  kernel_registry.cc
  tensor_utils.cc
  utils/type_info.cc)
