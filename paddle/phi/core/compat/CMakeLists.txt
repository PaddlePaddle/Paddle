cc_library(
  arg_map_context
  SRCS arg_map_context.cc
  DEPS phi_enforce)
cc_library(
  op_utils
  SRCS op_utils.cc
  DEPS arg_map_context enforce)

set(convert_utils_deps data_type place op_utils phi_backends)

<<<<<<< HEAD
if(WITH_MKLDNN)
  set(convert_utils_deps ${convert_utils_deps} mkldnn)
endif()

=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
cc_library(
  convert_utils
  SRCS convert_utils.cc
  DEPS ${convert_utils_deps})
