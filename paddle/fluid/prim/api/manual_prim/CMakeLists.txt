add_subdirectory(utils)
if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    manual_eager_prim_api
    SRCS eager_prim_api.cc
    DEPS final_dygraph_function eager_utils)
endif()
cc_library(
  manual_static_prim_api
  SRCS static_prim_api.cc
  DEPS proto_desc static_utils)
