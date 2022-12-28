# compatible utils used for fluid op system
add_subdirectory(compat)

if(WITH_GPU)
  proto_library(external_error_proto SRCS external_error.proto)
endif()

cc_library(
  flags
  SRCS flags.cc
  DEPS gflags)

cc_library(errors SRCS errors.cc)
set(phi_enforce_deps errors flags)
if(WITH_GPU)
  set(phi_enforce_deps ${phi_enforce_deps} external_error_proto)
endif()
cc_library(
  phi_enforce
  SRCS enforce.cc
  DEPS ${phi_enforce_deps})

if(WITH_XPU)
  cc_library(
    kernel_factory
    SRCS kernel_factory.cc
    DEPS phi_enforce convert_utils phi_backends)
else()
  cc_library(
    kernel_factory
    SRCS kernel_factory.cc
    DEPS phi_enforce convert_utils)
endif()
cc_library(
  kernel_context
  SRCS kernel_context.cc
  DEPS phi_enforce phi_backends)

cc_library(
  ddim
  SRCS ddim.cc
  DEPS phi_enforce)
cc_library(
  tensor_base
  SRCS tensor_base.cc allocator.cc
  DEPS phi_enforce)
cc_library(
  tensor_meta
  SRCS tensor_meta.cc
  DEPS phi_enforce)
cc_library(
  lod_utils
  SRCS lod_utils.cc
  DEPS phi_enforce)
cc_library(
  threadpool
  SRCS threadpool.cc
  DEPS phi_enforce)

cc_library(
  dense_tensor
  SRCS dense_tensor.cc dense_tensor_impl.cc
  DEPS convert_utils tensor_meta tensor_base ddim)

target_link_libraries(dense_tensor malloc)

cc_library(
  sparse_coo_tensor
  SRCS sparse_coo_tensor.cc
  DEPS tensor_meta tensor_base)
cc_library(
  sparse_csr_tensor
  SRCS sparse_csr_tensor.cc
  DEPS dense_tensor tensor_base)
cc_library(
  string_tensor
  SRCS string_tensor.cc
  DEPS convert_utils tensor_meta tensor_base)

cc_library(
  tensor_array
  SRCS tensor_array.cc
  DEPS dense_tensor tensor_base)

cc_library(
  extended_tensor
  SRCS extended_tensor.cc
  DEPS tensor_base)

cc_library(
  meta_tensor
  SRCS meta_tensor.cc
  DEPS tensor_base tensor_meta dense_tensor)
cc_library(
  infermeta_utils
  SRCS infermeta_utils.cc
  DEPS meta_tensor)
cc_library(
  selected_rows
  SRCS selected_rows_impl.cc selected_rows.cc
  DEPS tensor_base dense_tensor phi_enforce ddim)
cc_library(
  phi_device_context
  SRCS device_context.cc
  DEPS dense_tensor selected_rows)
cc_library(
  serialization
  SRCS serialization.cc
  DEPS version tensor phi_device_context)

cc_library(
  custom_kernel
  SRCS custom_kernel.cc
  DEPS kernel_factory)

# Will remove once we implemented MKLDNN_Tensor
if(WITH_MKLDNN)
  add_dependencies(dense_tensor mkldnn)
  add_dependencies(tensor_base mkldnn)
endif()

if(WITH_GPU)
  nv_library(
    phi_tensor_utils
    SRCS tensor_utils.cc
    DEPS phi_backends dense_tensor selected_rows malloc memcpy device_context)
elseif(WITH_ROCM)
  hip_library(
    phi_tensor_utils
    SRCS tensor_utils.cc
    DEPS phi_backends dense_tensor selected_rows malloc memcpy device_context)
elseif(WITH_XPU_KP)
  xpu_library(
    phi_tensor_utils
    SRCS tensor_utils.cc
    DEPS phi_backends dense_tensor selected_rows malloc memcpy device_context)
else()
  cc_library(
    phi_tensor_utils
    SRCS tensor_utils.cc
    DEPS dense_tensor selected_rows malloc memcpy device_context phi_backends)
endif()
