file(GLOB_RECURSE DRR_SRCS "*.cc")

set(op_creator_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/op_creator_drr_gen.py
)
set(op_compat_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/ops/yaml/op_compat.yaml)

set(parsed_op_dir ${PADDLE_BINARY_DIR}/paddle/phi/ops/yaml)

set(op_forward_yaml_file ${parsed_op_dir}/ops.parsed.yaml)
set(op_backward_yaml_file ${parsed_op_dir}/backward.parsed.yaml)
set(fused_op_forward_yaml_file ${parsed_op_dir}/fused_ops.parsed.yaml)
set(fused_op_backward_yaml_file ${parsed_op_dir}/fused_backward.parsed.yaml)

set(op_yaml_file_1 ${parsed_op_dir}/inconsistent/static_ops.parsed.yaml)
set(op_yaml_file_2 ${parsed_op_dir}/inconsistent/static_backward.parsed.yaml)
set(op_yaml_file_3 ${parsed_op_dir}/inconsistent/update_ops.parsed.yaml)

set(op_yaml_files
    ${op_forward_yaml_file},${op_backward_yaml_file},${fused_op_forward_yaml_file},${fused_op_backward_yaml_file},${op_yaml_file_1},${op_yaml_file_2},${op_yaml_file_3}
)

set(pd_op_creator_file
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/drr/src/pd_op_factory_generated.cc)
set(pd_op_creator_file_tmp ${pd_op_creator_file}.tmp)

set(pd_dialect_name pd_op)

add_custom_command(
  OUTPUT ${pd_op_creator_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_creator_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --dialect_name
    ${pd_dialect_name} --op_creator_file ${pd_op_creator_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${pd_op_creator_file_tmp}
          ${pd_op_creator_file}
  COMMENT "copy_if_different ${pd_op_creator_file}"
  DEPENDS ${op_creator_gen_file}
          ${op_forward_yaml_file}
          ${op_backward_yaml_file}
          ${op_compat_yaml_file}
          ${op_yaml_file_1}
          ${op_yaml_file_2}
          op_dialect
  VERBATIM)

set(DRR_SRCS ${DRR_SRCS} ${pd_op_creator_file})

if(WITH_CINN)
  set(cinn_op_yaml_file
      ${PADDLE_BINARY_DIR}/paddle/cinn/hlir/dialect/generated/ops.parsed.yaml)

  set(cinn_op_yaml_source_file
      ${PADDLE_SOURCE_DIR}/paddle/cinn/hlir/dialect/operator/ir/ops.yaml)

  set(cinn_op_creator_file
      ${PADDLE_BINARY_DIR}/paddle/fluid/pir/drr/src/cinn_op_factory_generated.cc
  )
  set(cinn_op_creator_file_tmp ${cinn_op_creator_file}.tmp)
  set(cinn_dialect_name cinn_op)

  add_custom_command(
    OUTPUT ${cinn_op_creator_file}
    COMMAND
      ${PYTHON_EXECUTABLE} ${op_creator_gen_file} --op_yaml_files
      ${cinn_op_yaml_file} --op_compat_yaml_file ${op_compat_yaml_file}
      --dialect_name ${cinn_dialect_name} --op_creator_file
      ${cinn_op_creator_file_tmp}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${cinn_op_creator_file_tmp}
            ${cinn_op_creator_file}
    COMMENT "copy_if_different ${cinn_op_creator_file}"
    DEPENDS ${op_creator_gen_file} ${op_compat_yaml_file}
            ${cinn_op_yaml_source_file} op_dialect cinn_op_dialect
    VERBATIM)
  set(CINN_SOURCE_FILE ${cinn_op_creator_file})

  set(CINN_DEPS cinn_op_dialect)

  set(DRR_SRCS ${DRR_SRCS} ${CINN_SOURCE_FILE})
endif()

if(WITH_ONEDNN)
  set(onednn_dialect_name onednn_op)
  set(pir_op_onednn_yaml
      ${parsed_op_dir}/inconsistent/onednn_static.parsed.yaml)
  set(pd_ops_onednn_extra_yaml_file
      ${PADDLE_SOURCE_DIR}/paddle/phi/ops/yaml/inconsistent/onednn_ops_extra.yaml
  )
  set(onednn_op_creator_file
      ${PADDLE_BINARY_DIR}/paddle/fluid/pir/drr/src/onednn_op_factory_generated.cc
  )
  set(onednn_op_creator_file_tmp ${onednn_op_creator_file}.tmp)

  add_custom_command(
    OUTPUT ${onednn_op_creator_file}
    COMMAND
      ${PYTHON_EXECUTABLE} ${op_creator_gen_file} --op_yaml_files
      ${op_yaml_files} --op_compat_yaml_file ${op_compat_yaml_file}
      --dialect_name ${onednn_dialect_name} --op_creator_file
      ${onednn_op_creator_file_tmp} --onednn_yaml_file ${pir_op_onednn_yaml}
      --ops_onednn_extra_yaml_file ${pd_ops_onednn_extra_yaml_file}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${onednn_op_creator_file_tmp}
            ${onednn_op_creator_file}
    COMMENT "copy_if_different ${onednn_op_creator_file}"
    DEPENDS ${op_creator_gen_file}
            ${op_forward_yaml_file}
            ${op_backward_yaml_file}
            ${op_compat_yaml_file}
            ${op_yaml_file_1}
            ${op_yaml_file_2}
            op_dialect
    VERBATIM)

  set(ONEDNN_SOURCE_FILE ${onednn_op_creator_file})

  set(DRR_SRCS ${DRR_SRCS} ${ONEDNN_SOURCE_FILE})
endif()

cc_library(
  drr
  SRCS ${DRR_SRCS}
  DEPS op_dialect_vjp ${CINN_DEPS} pir pir_general_functions)
