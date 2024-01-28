file(GLOB DRR_SRCS "*.cc" "include/*.cc")

set(op_creator_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/op_creator_drr_gen.py
)
set(op_compat_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)
set(op_forward_yaml_file1
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml
)

set(op_forward_yaml_file3
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/update_ops.parsed.yaml
)
set(op_backward_yaml_file1
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/backward_ops.parsed.yaml
)
set(fused_op_forward_yaml_file
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_ops.parsed.yaml
)
set(fused_op_backward_yaml_file
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_backward.parsed.yaml
)

set(cinn_op_yaml_file
    ${PADDLE_BINARY_DIR}/paddle/cinn/hlir/dialect/generated/ops.parsed.yaml)

set(cinn_op_yaml_source_file
    ${PADDLE_SOURCE_DIR}/paddle/cinn/hlir/dialect/operator/ir/ops.yaml)

set(parsed_op_dir
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/generated)

set(op_yaml_file3 ${parsed_op_dir}/ops.parsed.yaml)
set(op_yaml_file4 ${parsed_op_dir}/ops_backward.parsed.yaml)
set(op_yaml_file5 ${parsed_op_dir}/update_ops.parsed.yaml)

set(op_yaml_files
    ${op_forward_yaml_file1},${op_backward_yaml_file1},${fused_op_forward_yaml_file},${fused_op_backward_yaml_file},${op_yaml_file3},${op_yaml_file4},${op_yaml_file5}
)

set(op_creator_file
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/drr/ir_op_factory_generated.cc)
set(op_creator_file_tmp ${op_creator_file}.tmp)

set(dialect_name pd_op)

set(cinn_op_creator_file
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/drr/cinn_op_factory_generated.cc)
set(cinn_op_creator_file_tmp ${cinn_op_creator_file}.tmp)

set(cinn_dialect_name cinn_op)

add_custom_command(
  OUTPUT ${op_creator_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_creator_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --dialect_name ${dialect_name}
    --op_creator_file ${op_creator_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_creator_file_tmp}
          ${op_creator_file}
  COMMENT "copy_if_different ${op_creator_file}"
  DEPENDS ${op_creator_gen_file}
          ${op_forward_yaml_file1}
          ${op_backward_yaml_file1}
          ${op_compat_yaml_file}
          ${op_yaml_file3}
          ${op_yaml_file4}
          op_dialect
  VERBATIM)

if(WITH_CINN AND NOT CINN_ONLY)
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

endif()

cc_library(
  drr
  SRCS ${DRR_SRCS} ${op_creator_file} ${CINN_SOURCE_FILE}
  DEPS op_dialect_vjp ${CINN_DEPS} pir)
