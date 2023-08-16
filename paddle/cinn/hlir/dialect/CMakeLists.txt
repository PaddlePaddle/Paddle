# TODO(Aurelius84): new_ir_compiler depends on pd_dialect and could
# not found under CINN_ONLY mode
if(NOT CINN_ONLY)
  set(CINN_DIALECT_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/cinn/hlir/dialect")
  set(CINN_DIALECT_SOURCE_DIR "${PADDLE_BINARY_DIR}/paddle/cinn/hlir/dialect")

  # Generate cinn_dialect files defining op using op_gen_file
  set(op_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect/op_generator/op_gen.py
  )

  set(op_compat_yaml_file 
    ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml
  )

  set(op_forward_yaml_file1
    ${PADDLE_SOURCE_DIR}/paddle/cinn/hlir/dialect/cinn_ops.parsed.yaml
  )

  set(op_backward_yaml_file1
    ${PADDLE_SOURCE_DIR}/paddle/cinn/hlir/dialect/cinn_backward_ops.parsed.yaml
  )

  set(op_yaml_files
    ${op_forward_yaml_file1},${op_backward_yaml_file1}
  )
  set(op_namespace cinn,dialect)
  set(dialect_name cinn)
  set(op_header_file ${CINN_DIALECT_BINARY_DIR}/cinn_op.h)
  set(op_source_file ${CINN_DIALECT_BINARY_DIR}/cinn_op.cc)
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
    DEPENDS ${op_gen_file} ${op_forward_yaml_file1} 
            ${op_backward_yaml_file1} 
            ${op_compat_yaml_file}
    VERBATIM)

  cinn_cc_library(cinn_dialect SRCS runtime_dialect.cc jit_kernel_op.cc cinn_dialect.cc DEPS
                  pd_dialect)

  target_include_directories(cinn_dialect PRIVATE ${CINN_DIALECT_BINARY_DIR})
endif()
