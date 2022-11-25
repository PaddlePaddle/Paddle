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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
cc_library(
  convert_utils
  SRCS convert_utils.cc
  DEPS ${convert_utils_deps})
