file(GLOB DRR_SRCS "*.cc" "api/*.cc")

set(op_creator_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/op_creator_drr_gen.py
)
set(op_compat_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)
set(op_forward_yaml_file1
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml
)
set(op_forward_yaml_file2
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/legacy_ops.parsed.yaml
)
set(op_backward_yaml_file1
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/backward_ops.parsed.yaml
)
set(op_backward_yaml_file2
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/legacy_backward_ops.parsed.yaml
)
set(fused_op_forward_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_ops.parsed.yaml
)
set(fused_op_backward_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_backward.parsed.yaml
)

set(parsed_op_dir
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/generated)

set(op_yaml_file3 ${parsed_op_dir}/ops.parsed.yaml)
set(op_yaml_file4 ${parsed_op_dir}/ops_backward.parsed.yaml)

set(op_yaml_files
    ${op_forward_yaml_file1},${op_forward_yaml_file2},${op_backward_yaml_file1},${op_backward_yaml_file2},${fused_op_forward_yaml_file},${fused_op_backward_yaml_file},${op_yaml_file3},${op_yaml_file4}
)

set(op_creator_file
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/drr/ir_op_factory_generated.cc)
set(op_creator_file_tmp ${op_creator_file}.tmp)

set(dialect_name pd_op)

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
          ${op_forward_yaml_file2}
          ${op_backward_yaml_file1}
          ${op_backward_yaml_file2}
          ${op_compat_yaml_file}
          ${op_yaml_file3}
          ${op_yaml_file4}
          pd_op_dialect_op
  VERBATIM)

cc_library(
  drr
  SRCS ${DRR_SRCS} ${op_creator_file}
  DEPS pd_op_dialect pir)
