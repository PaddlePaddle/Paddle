if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    primitive_backend_eager_experimental
    SRCS eager_backend.cc
    DEPS final_dygraph_function eager_utils phi)
endif()
cc_library(
  primitive_backend_static_experimental
  SRCS static_backend.cc
  DEPS pd_dialect)
