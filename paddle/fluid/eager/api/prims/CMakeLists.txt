cc_library(
  static_global_utils
  SRCS static_global_utils.cc
  DEPS proto_desc)
cc_library(
  utils_static
  SRCS utils_static.cc
  DEPS proto_desc operator static_global_utils)
cc_library(
  static_prim_ops
  SRCS static_prim_ops.cc
  DEPS proto_desc utils_static)
