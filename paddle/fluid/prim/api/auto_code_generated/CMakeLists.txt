set(api_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml"
)
set(legacy_api_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/legacy_ops.parsed.yaml"
)
set(api_compat_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml")
set(api_prim_yaml_path "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/api.yaml")
set(api_version_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_version.yaml")
set(tmp_eager_prim_api_cc_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated_prim/eager_prim_api.cc.tmp"
)
set(tmp_static_prim_api_cc_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated_prim/static_prim_api.cc.tmp"
)
set(tmp_prim_api_h_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated_prim/prim_generated_api.h.tmp"
)
set(eager_prim_api_cc_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated_prim/eager_prim_api.cc"
)
set(static_prim_api_cc_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated_prim/static_prim_api.cc"
)
set(prim_api_h_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
)
set(static_prim_api_template_path
    "${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated/template/static_prim_api.cc.tpl"
)
set(eager_prim_api_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated/eager_gen.py)
set(static_prim_api_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated/static_gen.py
)
set(prim_tensor_operants_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py
)

message("Eager prim api code generator")
execute_process(
  WORKING_DIRECTORY
    ${CMAKE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated
  COMMAND
    ${PYTHON_EXECUTABLE} ${eager_prim_api_gen_file} --api_yaml_path
    ${legacy_api_yaml_path} ${api_yaml_path} --prim_api_header_path
    ${tmp_prim_api_h_path} --eager_prim_api_source_path
    ${tmp_eager_prim_api_cc_path} --api_prim_yaml_path ${api_prim_yaml_path}
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "Eager prim api generate failed, exiting.")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${tmp_prim_api_h_path} ${prim_api_h_path})
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${tmp_eager_prim_api_cc_path} ${eager_prim_api_cc_path})
message("copy tmp_xxx_prim_api to xxx_prim_api")

message("Static prim api code generator")
execute_process(
  WORKING_DIRECTORY
    ${CMAKE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated
  COMMAND
    ${PYTHON_EXECUTABLE} ${static_prim_api_gen_file} --api_phi_yaml_path
    ${api_yaml_path} --api_phi_legacy_yaml_path ${legacy_api_yaml_path}
    --api_compat_yaml_path ${api_compat_yaml_path} --api_version_yaml_path
    ${api_version_yaml_path} --api_prim_yaml_path ${api_prim_yaml_path}
    --template_path ${static_prim_api_template_path} --output_path
    ${tmp_static_prim_api_cc_path}
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "Static prim api generate failed, exiting.")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_static_prim_api_cc_path}
          ${static_prim_api_cc_path})
message("copy tmp_xxx_prim_api to xxx_prim_api")

set(eager_tensor_operants_cc_path
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/utils/eager/eager_tensor_operants.cc)
set(eager_tensor_operants_h_path
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/utils/eager/eager_tensor_operants.h)
set(static_tensor_operants_cc_path
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/utils/static/static_tensor_operants.cc
)
set(static_tensor_operants_h_path
    ${PADDLE_SOURCE_DIR}/paddle/fluid/prim/utils/static/static_tensor_operants.h
)
set(tmp_eager_tensor_operants_cc_path ${eager_tensor_operants_cc_path}.tmp)
set(tmp_eager_tensor_operants_h_path ${eager_tensor_operants_h_path}.tmp)
set(tmp_static_tensor_operants_cc_path ${static_tensor_operants_cc_path}.tmp)
set(tmp_static_tensor_operants_h_path ${static_tensor_operants_h_path}.tmp)
set(tensor_api_yaml_path
    ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/tensor_operants.yaml)

message("Prim tensor operants code generator")
execute_process(
  WORKING_DIRECTORY
    ${CMAKE_SOURCE_DIR}/paddle/fluid/prim/api/auto_code_generated
  COMMAND
    ${PYTHON_EXECUTABLE} ${prim_tensor_operants_gen_file} --api_yaml_path
    ${legacy_api_yaml_path} ${api_yaml_path} --eager_tensor_operants_header_path
    ${tmp_eager_tensor_operants_h_path} --eager_tensor_operants_source_path
    ${tmp_eager_tensor_operants_cc_path} --static_tensor_operants_header_path
    ${tmp_static_tensor_operants_h_path} --static_tensor_operants_source_path
    ${tmp_static_tensor_operants_cc_path} --api_prim_yaml_path
    ${tensor_api_yaml_path}
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "Prim tensor operants generate failed, exiting.")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${tmp_eager_tensor_operants_h_path} ${eager_tensor_operants_h_path})
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${tmp_eager_tensor_operants_cc_path} ${eager_tensor_operants_cc_path})
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${tmp_static_tensor_operants_h_path} ${static_tensor_operants_h_path})
execute_process(
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different ${tmp_static_tensor_operants_cc_path}
    ${static_tensor_operants_cc_path})
message("copy prim xxx_tensor_operants.tmp to xxx_tensor_operants")
