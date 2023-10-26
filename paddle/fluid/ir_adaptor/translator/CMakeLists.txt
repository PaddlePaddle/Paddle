set(PD_PROGRAM_TRANSLATOR_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(PD_PROGRAM_TRANSLATOR_BINARY_DIR
    "${PADDLE_BINARY_DIR}/paddle/fluid/ir_adaptor/translator/")

set(op_gen_file ${PD_PROGRAM_TRANSLATOR_SOURCE_DIR}/op_compat_gen.py)
set(op_compat_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)
set(op_compat_source_file ${PD_PROGRAM_TRANSLATOR_SOURCE_DIR}/op_compat_info.cc)
set(op_compat_templat_file
    ${PD_PROGRAM_TRANSLATOR_SOURCE_DIR}/op_compat_info.cc.j2)

add_custom_command(
  OUTPUT ${op_compat_source_file}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_file} --op_compat_yaml_file
          ${op_compat_yaml_file} --output_source_file ${op_compat_source_file}
  DEPENDS ${op_gen_file} ${op_compat_yaml_file} ${op_compat_templat_file}
  VERBATIM)

file(GLOB PD_PROGRAM_TRANSLATOR_SRCS "*.cc")

cc_library(
  program_translator
  SRCS ${PD_PROGRAM_TRANSLATOR_SRCS} ${op_compat_source_file}
  DEPS proto_desc pd_op_dialect pir framework_proto)
