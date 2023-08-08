set(PD_DIALECT_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect")
set(PD_DIALECT_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/fluid/ir/dialect")

# Generate pd_dialect files defining op using op_gen_file
set(op_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect/op_generator/op_gen.py)
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
set(op_yaml_file3 ${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect/pd_op.yaml)

set(op_yaml_files
    ${op_forward_yaml_file1},${op_forward_yaml_file2},${op_backward_yaml_file1},${op_backward_yaml_file2},${op_yaml_file3}
)
set(op_namespace paddle,dialect)
set(dialect_name pd)
set(op_header_file ${PD_DIALECT_BINARY_DIR}/pd_op.h)
set(op_source_file ${PD_DIALECT_BINARY_DIR}/pd_op.cc)
set(op_header_file_tmp ${op_header_file}.tmp)
set(op_source_file_tmp ${op_source_file}.tmp)

add_custom_command(
  OUTPUT ${op_header_file} ${op_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces ${op_namespace}
    --dialect_name ${dialect_name} --op_def_h_file ${op_header_file_tmp}
    --op_def_cc_file ${op_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_header_file_tmp}
          ${op_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_source_file_tmp}
          ${op_source_file}
  COMMENT "copy_if_different ${op_header_file} ${op_source_file}"
  DEPENDS ${op_gen_file} ${op_forward_yaml_file1} ${op_forward_yaml_file2}
          ${op_backward_yaml_file1} ${op_backward_yaml_file2}
          ${op_compat_yaml_file}
  VERBATIM)

# All source files of pd_dialect, except for the source file of op, which is generated in the compilation directory.
file(GLOB PD_DIALECT_SRCS "*.cc")

cc_library(
  pd_dialect
  SRCS ${PD_DIALECT_SRCS} ${op_source_file}
  DEPS framework_proto
       phi
       phi_utils
       pd_interface
       pd_trait
       ir
       primitive_vjp_experimental
       type_info)
target_include_directories(pd_dialect PRIVATE ${PD_DIALECT_BINARY_DIR})
