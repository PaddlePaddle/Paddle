if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    experimental_eager_primitive
    SRCS eager_primitive.cc
    DEPS final_dygraph_function eager_utils)
endif()
cc_library(
  experimental_static_primitive
  SRCS static_primitive.cc
  DEPS proto_desc static_utils)
