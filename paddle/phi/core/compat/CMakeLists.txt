cc_library(
  arg_map_context
  SRCS arg_map_context.cc
  DEPS phi_enforce)
cc_library(
  op_utils
  SRCS op_utils.cc
  DEPS arg_map_context enforce)
cc_library(
  get_kerneltype_forvar_utils
  SRCS get_kerneltype_forvar_utils.cc
  DEPS enforce)

set(convert_utils_deps data_type place op_utils phi_backends)

if(WITH_MKLDNN)
  set(convert_utils_deps ${convert_utils_deps} mkldnn)
endif()

cc_library(
  convert_utils
  SRCS convert_utils.cc
  DEPS ${convert_utils_deps})
