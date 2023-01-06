if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    eager_prim_api
    SRCS eager_prim_api.cc
    DEPS eager_utils)
  add_dependencies(eager_prim_api prim_api_codegen)
endif()
