set(eager_backend_files
    ${CMAKE_CURRENT_BINARY_DIR}/generated/generated_eager_backend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/manual/manual_eager_prim_backend.cc)
if(WITH_PYTHON OR NOT ON_INFER)
  cc_library(
    primitive_backend_eager_experimental
    SRCS ${eager_backend_files}
    DEPS final_dygraph_function eager_utils phi common)
endif()
set(static_backend_files
    ${CMAKE_CURRENT_BINARY_DIR}/generated/generated_static_backend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/manual/manual_static_backend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/manual/manual_static_prim_backend.cc)
cc_library(
  primitive_backend_static_experimental
  SRCS ${static_backend_files}
  DEPS op_dialect)
