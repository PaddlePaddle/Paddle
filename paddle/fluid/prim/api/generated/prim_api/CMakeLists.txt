# set(api_yaml_path
#     "${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/ops.yaml"
# )
# set(legacy_api_yaml_path
#     "${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_ops.yaml"
# )
set(api_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml"
)
set(legacy_api_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/legacy_ops.parsed.yaml"
)
set(tmp_eager_prim_api_cc_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated/prim_api/tmp_eager_prim_api.cc"
)
set(tmp_prim_api_h_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated/prim_api/tmp_prim_api.h"
)
set(eager_prim_api_cc_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated/prim_api/eager_prim_api.cc"
)
set(prim_api_h_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated/prim_api/prim_api.h")
set(prim_api_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated/prim_gen.py)
message("Prim Api CodeGen")
add_custom_command(
  OUTPUT ${eager_prim_api_cc_path} ${prim_api_h_path}
  COMMAND
    ${PYTHON_EXECUTABLE} ${prim_api_gen_file} --api_yaml_path
    ${legacy_api_yaml_path} ${api_yaml_path} --prim_api_header_path
    ${tmp_prim_api_h_path} --eager_prim_api_source_path
    ${tmp_eager_prim_api_cc_path}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_eager_prim_api_cc_path}
          ${eager_prim_api_cc_path}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_prim_api_h_path}
          ${prim_api_h_path}
  DEPENDS ${legacy_api_yaml_path} ${api_yaml_path} ${prim_api_gen_file}
  VERBATIM)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    eager_prim_api
    SRCS eager_prim_api.cc
    DEPS eager_utils)
endif()
