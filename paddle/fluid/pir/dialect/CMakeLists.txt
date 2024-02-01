set(PD_DIALECT_SOURCE_DIR
    "${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir")

# Generate pd_op_dialect files defining op using op_gen_file
set(op_gen_parsed_yaml_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/operators/generator/parse_op.py)

set(op_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/op_generator/op_gen.py)

set(op_compat_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)

# YAML files for defining operators
set(op_fwd_yaml
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml
)
set(op_bwd_yaml
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/backward_ops.parsed.yaml
)
set(fused_op_fwd_yaml
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_ops.parsed.yaml
)
set(fused_op_bwd_yaml
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/generator/parsed_ops/fused_backward.parsed.yaml
)
set(pir_op_fwd_src_yaml
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/ops.yaml)
set(pir_op_bwd_src_yaml
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/ops_backward.yaml)

set(pir_update_op_fwd_src_yaml
    ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/update_ops.yaml)
set(parsed_op_dir
    ${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir/generated)
set(pir_op_fwd_yaml ${parsed_op_dir}/ops.parsed.yaml)
set(pir_op_bwd_yaml ${parsed_op_dir}/ops_backward.parsed.yaml)
set(pir_update_op_fwd_yaml ${parsed_op_dir}/update_ops.parsed.yaml)

set(op_yaml_files
    ${op_fwd_yaml},${op_bwd_yaml},${fused_op_fwd_yaml},${fused_op_bwd_yaml},${pir_op_fwd_yaml},${pir_op_bwd_yaml},${pir_update_op_fwd_yaml}
)

# SRC files for defining operators
set(op_namespace paddle,dialect)
set(dialect_name pd_op)
set(op_header_file ${PD_DIALECT_SOURCE_DIR}/pd_op.h)
set(op_header_file_tmp ${op_header_file}.tmp)

set(op_info_file ${PD_DIALECT_SOURCE_DIR}/pd_op_info.cc)
set(op_info_file_tmp ${op_info_file}.tmp)

set(op_vjp_source_file ${PD_DIALECT_SOURCE_DIR}/pd_op_vjp.cc)
set(op_vjp_source_file_tmp ${op_vjp_source_file}.tmp)

set(op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_op.cc)
set(op_source_file_tmp ${op_source_file}.tmp)

set(bwd_op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_op_bwd.cc)
set(bwd_op_source_file_tmp ${bwd_op_source_file}.tmp)

set(fused_op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_op_fused.cc)
set(fused_op_source_file_tmp ${fused_op_source_file}.tmp)

set(bwd_fused_op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_op_fused_bwd.cc)
set(bwd_fused_op_source_file_tmp ${bwd_fused_op_source_file}.tmp)

set(pir_op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_pir_op.cc)
set(pir_op_source_file_tmp ${pir_op_source_file}.tmp)

set(pir_bwd_op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_pir_op_bwd.cc)
set(pir_bwd_op_source_file_tmp ${pir_bwd_op_source_file}.tmp)

set(pir_update_op_source_file ${PD_DIALECT_SOURCE_DIR}/pd_pir_op_update.cc)
set(pir_update_op_source_file_tmp ${pir_update_op_source_file}.tmp)

set(op_src_files_tmp
    ${op_source_file_tmp},${bwd_op_source_file_tmp},${fused_op_source_file_tmp},${bwd_fused_op_source_file_tmp},${pir_op_source_file_tmp},${pir_bwd_op_source_file_tmp},${pir_update_op_source_file_tmp}
)

set(op_vjp_src_file_tmp ${op_vjp_source_file_tmp})

# Auto code gen
execute_process(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
          ${pir_op_fwd_src_yaml} --output_path ${pir_op_fwd_yaml}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
          ${pir_update_op_fwd_src_yaml} --output_path ${pir_update_op_fwd_yaml}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
          ${pir_op_bwd_src_yaml} --output_path ${pir_op_bwd_yaml} --backward)

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces ${op_namespace}
    --dialect_name ${dialect_name} --op_def_h_file ${op_header_file_tmp}
    --op_info_file ${op_info_file_tmp} --op_def_cc_file ${op_src_files_tmp}
    --op_vjp_cc_file ${op_vjp_src_file_tmp})

set(generated_files_pd_op
    "${op_header_file}"
    "${op_info_file}"
    "${op_source_file}"
    "${op_vjp_source_file}"
    "${bwd_op_source_file}"
    "${fused_op_source_file}"
    "${bwd_fused_op_source_file}"
    "${pir_op_source_file}"
    "${pir_bwd_op_source_file}"
    "${pir_update_op_source_file}")

if(WITH_MKLDNN)
  set(pir_op_onednn_yaml ${parsed_op_dir}/onednn.parsed.yaml)

  set(pd_onednn_op_yaml_file
      ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/onednn.yaml)

  set(pd_ops_onednn_extra_yaml_file
      ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/dialect/operator/ir/ops_onednn_extra.yaml
  )

  set(op_onednn_info_file ${PD_DIALECT_SOURCE_DIR}/onednn_op_info.cc)
  set(op_onednn_info_file_tmp ${op_onednn_info_file}.tmp)

  set(onednn_op_namespace paddle,onednn,dialect)
  set(onednn_dialect_name onednn_op)
  set(onednn_op_header_file ${PD_DIALECT_SOURCE_DIR}/onednn_op.h)
  set(onednn_op_source_file ${PD_DIALECT_SOURCE_DIR}/onednn_op.cc)
  set(onednn_op_header_file_tmp ${onednn_op_header_file}.tmp)
  set(onednn_op_source_file_tmp ${onednn_op_source_file}.tmp)

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
            ${pd_onednn_op_yaml_file} --output_path ${pir_op_onednn_yaml})

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${op_gen_file} --op_yaml_files ${op_yaml_files}
      --op_compat_yaml_file ${op_compat_yaml_file} --namespaces
      ${onednn_op_namespace} --dialect_name ${onednn_dialect_name}
      --op_def_h_file ${onednn_op_header_file_tmp} --op_info_file
      ${op_onednn_info_file_tmp} --op_def_cc_file ${onednn_op_source_file_tmp}
      --onednn_yaml_file ${pir_op_onednn_yaml} --ops_onednn_extra_yaml_file
      ${pd_ops_onednn_extra_yaml_file})

  set(generated_files_onednn_pd_op
      "${onednn_op_header_file}" "${onednn_op_source_file}"
      "${op_onednn_info_file}")
endif()
set(api_gen_yaml_files
    ${op_fwd_yaml},${op_bwd_yaml},${pir_op_fwd_yaml},${pir_op_bwd_yaml},${pir_update_op_fwd_yaml}
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

set(generated_files_pir
    ${generated_files_pd_op} ${generated_files_onednn_pd_op}
    ${generated_files_pd_api} ${generated_files_python_c}
    ${generated_files_ops_api})
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
file(
  GLOB_RECURSE
  op_dialect_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/kernel/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/operator/interface/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/operator/trait/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/operator/transforms/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/operator/utils/*.cc")

set(op_dialect_srcs
    ${op_dialect_srcs}
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/api_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/control_flow_op.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/ir_meta_tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/ir_selected_rows.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/ir_tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_api.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_op.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/op_attribute.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/op_type.cc
    ${op_info_file}
    ${op_source_file}
    ${bwd_op_source_file}
    ${fused_op_source_file}
    ${bwd_fused_op_source_file}
    ${pir_op_source_file}
    ${pir_bwd_op_source_file}
    ${pir_update_op_source_file}
    ${api_source_file})

if(WITH_MKLDNN)
  set(op_dialect_srcs ${op_dialect_srcs} ${onednn_op_source_file}
                      ${op_onednn_info_file})
endif()

set(op_dialect_deps phi common pir type_info string_helper)

cc_library(
  op_dialect
  SRCS ${op_dialect_srcs}
  DEPS ${op_dialect_deps})

#Note(risemeup1):compile some *.cc files which depend on primitive_vjp_experimental into op_dialect_vjp.a/lib
set(op_decomp_source_file ${PD_DIALECT_SOURCE_DIR}/op_decomp.cc)
set(op_dialect_vjp_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_op_decomp.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/manual_op_vjp.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/op_dialect.cc
    ${op_decomp_source_file}
    ${op_vjp_source_file}
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/base/decomp_trans.cc)

if(WITH_MKLDNN)
  set(op_dialect_vjp_srcs
      ${op_dialect_vjp_srcs}
      ${CMAKE_CURRENT_SOURCE_DIR}/operator/ir/op_onednn_dialect.cc)
endif()

set(op_dialect_vjp_deps primitive_vjp_experimental op_dialect)

cc_library(
  op_dialect_vjp
  SRCS ${op_dialect_vjp_srcs}
  DEPS ${op_dialect_vjp_deps})

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  set_target_properties(op_dialect PROPERTIES COMPILE_FLAGS
                                              "-Wno-maybe-uninitialized")
endif()
