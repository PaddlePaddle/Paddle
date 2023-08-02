if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    experimental_eager_primitive_backend
    SRCS eager_backend.cc
    DEPS final_dygraph_function eager_utils phi)
endif()
cc_library(
  experimental_static_primitive_backend
  SRCS static_backend.cc
  DEPS pd_dialect phi)
