cc_library(
  static_utils
  SRCS static_utils.cc
  DEPS proto_desc operator static_global_utils)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    eager_utils
    SRCS eager_utils.cc
    DEPS final_dygraph_function)
endif()
