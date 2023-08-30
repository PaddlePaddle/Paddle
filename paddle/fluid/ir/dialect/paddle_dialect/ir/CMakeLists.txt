set(PD_DIALECT_BINARY_DIR
    "${PADDLE_BINARY_DIR}/paddle/fluid/ir/dialect/paddle_dialect/ir")

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
set(fused_op_forward_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_ops.parsed.yaml
)
set(fused_op_backward_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_backward.parsed.yaml
)
set(op_yaml_file3
    ${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.yaml)

set(op_yaml_files
    ${op_forward_yaml_file1},${op_forward_yaml_file2},${op_backward_yaml_file1},${op_backward_yaml_file2},${fused_op_forward_yaml_file},${fused_op_backward_yaml_file},${op_yaml_file3}
)
set(op_namespace paddle,dialect)
set(dialect_name pd)
set(op_header_file ${PD_DIALECT_BINARY_DIR}/pd_op.h)
set(op_source_file ${PD_DIALECT_BINARY_DIR}/pd_op.cc)
set(op_header_file_tmp ${op_header_file}.tmp)
set(op_source_file_tmp ${op_source_file}.tmp)

set(op_vjp_source_file ${PD_DIALECT_BINARY_DIR}/pd_op_vjp.cc)
set(op_vjp_source_file_tmp ${op_vjp_source_file}.tmp)

add_custom_command(
  OUTPUT ${op_header_file} ${op_source_file} ${op_vjp_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces ${op_namespace}
    --dialect_name ${dialect_name} --op_def_h_file ${op_header_file_tmp}
    --op_def_cc_file ${op_source_file_tmp} --op_vjp_cc_file
    ${op_vjp_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_header_file_tmp}
          ${op_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_source_file_tmp}
          ${op_source_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_vjp_source_file_tmp}
          ${op_vjp_source_file}
  COMMENT
    "copy_if_different ${op_header_file} ${op_source_file} ${op_vjp_source_file}"
  DEPENDS ${op_gen_file} ${op_forward_yaml_file1} ${op_forward_yaml_file2}
          ${op_backward_yaml_file1} ${op_backward_yaml_file2}
          ${op_compat_yaml_file}
  VERBATIM)

set(api_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect/op_generator/api_gen.py)
set(api_header_file ${PD_DIALECT_BINARY_DIR}/pd_api.h)
set(api_source_file ${PD_DIALECT_BINARY_DIR}/pd_api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)

add_custom_command(
  OUTPUT ${api_header_file} ${api_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${api_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces ${op_namespace}
    --api_def_h_file ${api_header_file_tmp} --api_def_cc_file
    ${api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_header_file_tmp}
          ${api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_source_file_tmp}
          ${api_source_file}
  COMMENT "copy_if_different ${api_header_file} ${api_source_file}"
  DEPENDS ${api_gen_file} ${op_forward_yaml_file1} ${op_forward_yaml_file2}
          ${op_backward_yaml_file1} ${op_backward_yaml_file2}
          ${op_compat_yaml_file}
  VERBATIM)

set(python_c_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect/op_generator/python_c_gen.py)
set(python_c_header_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/static_op_function.h)
set(python_c_source_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/static_op_function.cc)
set(python_c_header_file_tmp ${python_c_header_file}.tmp)
set(python_c_source_file_tmp ${python_c_source_file}.tmp)

add_custom_command(
  OUTPUT ${python_c_header_file} ${python_c_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${python_c_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces "paddle,pybind"
    --python_c_def_h_file ${python_c_header_file_tmp} --python_c_def_cc_file
    ${python_c_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${python_c_header_file_tmp}
          ${python_c_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${python_c_source_file_tmp}
          ${python_c_source_file}
  COMMENT "copy_if_different ${python_c_header_file} ${python_c_source_file}"
  DEPENDS ${python_c_gen_file} ${op_forward_yaml_file1} ${op_forward_yaml_file2}
          ${op_backward_yaml_file1} ${op_backward_yaml_file2}
          ${op_compat_yaml_file}
  VERBATIM)

add_custom_target(static_op_function_gen ALL DEPENDS ${python_c_header_file}
                                                     ${python_c_source_file})

cc_library(
  pd_dialect_core
  SRCS pd_attribute.cc pd_type.cc
  DEPS phi pd_interface pd_trait type_info)
cc_library(
  pd_dialect_op
  SRCS ${op_source_file} pd_manual_op.cc
  DEPS pd_dialect_core)
cc_library(
  api_builder
  SRCS api_builder.cc
  DEPS ir_core)
cc_library(
  pd_dialect_api
  SRCS ${api_source_file} pd_manual_api.cc
  DEPS api_builder pd_dialect_op)

target_include_directories(pd_dialect_api PRIVATE ${PD_DIALECT_BINARY_DIR})

cc_library(
  pd_dialect
  SRCS pd_dialect.cc pd_manual_op_vjp.cc ${op_vjp_source_file}
  DEPS pd_dialect_api param_to_variable primitive_vjp_experimental
       pd_dialect_utils op_yaml_info_parser)
