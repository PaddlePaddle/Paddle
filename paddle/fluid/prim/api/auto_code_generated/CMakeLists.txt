
set(api_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/ops.yaml"
)
set(legacy_api_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_ops.yaml"
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
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated/prim_api/prim_api.h"
)

message("Prim Api CodeGen")
add_custom_target(
  prim_api_codegen
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated/prim_gen.py
    --api_yaml_path ${legacy_api_yaml_path} ${api_yaml_path}
    --prim_api_header_path ${tmp_prim_api_h_path}
    --eager_prim_api_source_path ${tmp_eager_prim_api_cc_path}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_eager_prim_api_cc_path}
          ${eager_prim_api_cc_path}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_prim_api_h_path}
          ${prim_api_h_path}
  VERBATIM)


