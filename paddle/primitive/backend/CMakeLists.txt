if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    primitive_backend_eager_experimental
    SRCS eager_backend.cc
    DEPS final_dygraph_function eager_utils phi)
endif()
