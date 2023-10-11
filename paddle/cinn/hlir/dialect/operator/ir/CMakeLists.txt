# TODO(Aurelius84): new_ir_compiler depends on pd_op_dialect and could
# not found under CINN_ONLY mode
if(NOT CINN_ONLY)
  set(CINN_DIALECT_BINARY_DIR
      "${PADDLE_BINARY_DIR}/paddle/cinn/hlir/dialect/operator/ir")

  # Generate cinn_op_dialect files defining op using op_gen_file
  set(cinn_op_gen_parsed_yaml_file
      ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parse_op.py)

  set(cinn_op_gen_file
      ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/op_gen.py)

  set(cinn_op_compat_yaml_file
      ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)

  set(cinn_op_yaml_file
      ${PADDLE_SOURCE_DIR}/paddle/cinn/hlir/dialect/operator/ir/ops.yaml)

  set(parsed_op_dir ${PADDLE_SOURCE_DIR}/paddle/cinn/hlir/dialect/generated)

  set(cinn_op_parsed_yaml_file ${parsed_op_dir}/ops.parsed.yaml)

  set(cinn_op_parsed_yaml_files ${cinn_op_parsed_yaml_file})

  set(cinn_op_namespace cinn,dialect)
  set(cinn_op_dialect_name cinn_op)
  set(cinn_op_header_file ${CINN_DIALECT_BINARY_DIR}/cinn_op.h)
  set(cinn_op_source_file ${CINN_DIALECT_BINARY_DIR}/cinn_op.cc)
  set(cinn_op_header_file_tmp ${cinn_op_header_file}.tmp)
  set(cinn_op_source_file_tmp ${cinn_op_source_file}.tmp)

  add_custom_command(
    OUTPUT ${cinn_op_parsed_yaml_file}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir}
    COMMAND ${PYTHON_EXECUTABLE} ${cinn_op_gen_parsed_yaml_file} --op_yaml_path
            ${cinn_op_yaml_file} --output_path ${cinn_op_parsed_yaml_file}
    VERBATIM)

  add_custom_command(
    OUTPUT ${cinn_op_header_file} ${cinn_op_source_file}
    COMMAND
      ${PYTHON_EXECUTABLE} ${cinn_op_gen_file} --op_yaml_files
      ${cinn_op_parsed_yaml_files} --op_compat_yaml_file
      ${cinn_op_compat_yaml_file} --namespaces ${cinn_op_namespace}
      --dialect_name ${cinn_op_dialect_name} --op_def_h_file
      ${cinn_op_header_file_tmp} --op_def_cc_file ${cinn_op_source_file_tmp}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${cinn_op_header_file_tmp}
            ${cinn_op_header_file}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${cinn_op_source_file_tmp}
            ${cinn_op_source_file}
    DEPENDS ${cinn_op_gen_file} ${cinn_op_parsed_yaml_file}
            ${cinn_op_compat_yaml_file}
    VERBATIM)

  cinn_cc_library(
    cinn_op_dialect
    SRCS
    op_dialect.cc
    ${cinn_op_source_file}
    manual_op.cc
    op_attribute.cc
    DEPS
    pd_op_dialect)

  target_include_directories(cinn_op_dialect PRIVATE ${CINN_DIALECT_BINARY_DIR})
endif()
