if(WITH_PYTHON OR NOT ON_INFER)
  cc_library(
    primitive_eager_utils_experimental
    SRCS decomp_eager_utils.cc
    DEPS phi common)
endif()
cc_library(
  primitive_static_utils_experimental
  SRCS decomp_static_utils.cc
  DEPS phi common op_dialect)
