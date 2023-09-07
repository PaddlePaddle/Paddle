cc_library(
  static_global_utils
  SRCS static_global_utils.cc
  DEPS proto_desc)

cc_library(
  static_tensor_operants
  SRCS static_tensor_operants.cc
  DEPS static_prim_api primitive_backend_static_experimental)
