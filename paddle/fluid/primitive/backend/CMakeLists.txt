set(eager_backend_files
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/backend/generated/generated_eager_backend.cc
)
if(WITH_PYTHON OR NOT ON_INFER)
  cc_library(
    primitive_backend_eager_experimental
    SRCS ${eager_backend_files}
    DEPS final_dygraph_function eager_utils phi)
endif()
set(static_backend_files
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/backend/generated/generated_static_backend.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/backend/manual/manual_static_backend.cc
)
cc_library(
  primitive_backend_static_experimental
  SRCS ${static_backend_files}
  DEPS pd_dialect)
