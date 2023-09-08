if(WITH_PYTHON OR NOT ON_INFER)
  cc_library(
    primitive_eager_utils_experimental
    SRCS eager_utils.cc
    DEPS phi common_infer_shape_functions)
endif()
cc_library(
  primitive_static_utils_experimental
  SRCS static_utils.cc
  DEPS phi common_infer_shape_functions)
