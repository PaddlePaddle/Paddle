set(PD_DIALECT_SOURCE_DIR
    "${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir")

# Generate pd_op_dialect files defining op using op_gen_file
set(op_gen_parsed_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parse_op.py)

set(op_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/op_gen.py)
set(op_compat_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)
set(op_forward_yaml_file1
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml
)
set(op_backward_yaml_file1
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/backward_ops.parsed.yaml
)
set(fused_op_forward_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_ops.parsed.yaml
)
set(fused_op_backward_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_backward.parsed.yaml
)

set(pd_op_forward_yaml_file1
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/ops.yaml)

set(pd_op_forward_yaml_file2
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/update_ops.yaml)

set(pd_op_backward_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/ops_backward.yaml)

set(parsed_op_dir
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/generated)

set(op_yaml_file3 ${parsed_op_dir}/ops.parsed.yaml)
set(op_yaml_file4 ${parsed_op_dir}/ops_backward.parsed.yaml)
set(op_yaml_file5 ${parsed_op_dir}/update_ops.parsed.yaml)

set(op_yaml_files
    ${op_forward_yaml_file1},${op_backward_yaml_file1},${fused_op_forward_yaml_file},${fused_op_backward_yaml_file},${op_yaml_file3},${op_yaml_file4},${op_yaml_file5}
)
set(op_namespace paddle,dialect)
set(dialect_name pd_op)
set(op_header_file ${PD_DIALECT_SOURCE_DIR}/pd_op.h)
set(op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_op.cc)
set(op_header_file_tmp ${op_header_file}.tmp)
set(op_source_file_tmp ${op_source_file}.tmp)

set(op_vjp_source_file ${PD_DIALECT_SOURCE_DIR}/pd_op_vjp.cc)
set(op_decomp_source_file ${PD_DIALECT_SOURCE_DIR}/op_decomp.cc)
set(op_vjp_source_file_tmp ${op_vjp_source_file}.tmp)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
          ${pd_op_forward_yaml_file1} --output_path ${op_yaml_file3}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
          ${pd_op_forward_yaml_file2} --output_path ${op_yaml_file5}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
          ${pd_op_backward_yaml_file} --output_path ${op_yaml_file4} --backward)

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces ${op_namespace}
    --dialect_name ${dialect_name} --op_def_h_file ${op_header_file_tmp}
    --op_def_cc_file ${op_source_file_tmp} --op_vjp_cc_file
    ${op_vjp_source_file_tmp})

set(generated_files_pd_op "${op_header_file}" "${op_source_file}"
                          "${op_vjp_source_file}")

set(api_gen_yaml_files
    ${op_forward_yaml_file1},${op_backward_yaml_file1},${op_yaml_file3},${op_yaml_file4},${op_yaml_file5}
)
set(api_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/api_gen.py)
set(api_header_file ${PD_DIALECT_SOURCE_DIR}/pd_api.h)
set(api_source_file ${PD_DIALECT_SOURCE_DIR}/pd_api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${api_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces ${op_namespace}
    --api_def_h_file ${api_header_file_tmp} --api_def_cc_file
    ${api_source_file_tmp})

set(generated_files_pd_api "${api_header_file}" "${api_source_file}")

set(python_c_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/python_c_gen.py)
set(python_c_header_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/static_op_function.h)
set(python_c_source_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/static_op_function.cc)
set(python_c_header_file_tmp ${python_c_header_file}.tmp)
set(python_c_source_file_tmp ${python_c_source_file}.tmp)

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${python_c_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces "paddle,pybind"
    --python_c_def_h_file ${python_c_header_file_tmp} --python_c_def_cc_file
    ${python_c_source_file_tmp})

set(generated_files_python_c "${python_c_header_file}"
                             "${python_c_source_file}")

set(ops_api_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/ops_api_gen.py)
set(ops_api_source_file ${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/ops_api.cc)
set(ops_api_source_file_tmp ${ops_api_source_file}.tmp)

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${ops_api_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces "paddle,pybind"
    --ops_api_file ${ops_api_source_file_tmp})

set(generated_files_ops_api "${ops_api_source_file}")

set(generated_files_pir ${generated_files_pd_op} ${generated_files_pd_api}
                        ${generated_files_python_c} ${generated_files_ops_api})
foreach(generated_file ${generated_files_pir})
  if(EXISTS "${generated_file}.tmp" AND EXISTS "${generated_file}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${generated_file}.tmp" "${generated_file}")
    message("copy if different ${generated_file}.tmp ${generated_file}")
  elseif(EXISTS "${generated_file}.tmp")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${generated_file}.tmp"
                            "${generated_file}")
    message("copy ${generated_file}.tmp ${generated_file}")
  endif()
endforeach()

add_custom_target(ops_api_gen ALL DEPENDS ${ops_api_source_file})

#Note(risemeup1):compile some *.cc files which do not depend on primitive_vjp_experimental into op_dialect.a/lib
file(GLOB_RECURSE op_dialect_srcs "*.cc")
list(
  REMOVE_ITEM
  op_dialect_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_op_decomp.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_op_vjp.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/op_dialect.cc)

set(op_dialect_srcs ${op_dialect_srcs} ${op_source_file} ${api_source_file})

set(op_dialect_deps phi common pir type_info string_helper)

cc_library(
  op_dialect
  SRCS ${op_dialect_srcs}
  DEPS ${op_dialect_deps})

#Note(risemeup1):compile some *.cc files which depend on primitive_vjp_experimental into op_dialect_vjp.a/lib
set(op_dialect_vjp_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_op_decomp.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_op_vjp.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/op_dialect.cc
    ${op_decomp_source_file}
    ${op_vjp_source_file})
set(op_dialect_vjp_deps primitive_vjp_experimental op_dialect)

cc_library(
  op_dialect_vjp
  SRCS ${op_dialect_vjp_srcs}
  DEPS ${op_dialect_vjp_deps})

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  set_target_properties(op_dialect PROPERTIES COMPILE_FLAGS
                                              "-Wno-maybe-uninitialized")
endif()
