add_subdirectory(utils)

if(WITH_GPU)
  nv_library(
    phi_tensor_raw
    SRCS tensor.cc
    DEPS tensor_base dense_tensor phi_api_utils phi_enforce context_pool)
elseif(WITH_ROCM)
  hip_library(
    phi_tensor_raw
    SRCS tensor.cc
    DEPS tensor_base dense_tensor phi_api_utils phi_enforce context_pool)
else()
  cc_library(
    phi_tensor_raw
    SRCS tensor.cc
    DEPS tensor_base dense_tensor phi_api_utils phi_enforce context_pool)
endif()

set(api_gen_base ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/api_base.py)

# forward api file
set(api_gen_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/api_gen.py)
set(api_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/ops.yaml)
set(legacy_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_ops.yaml)
set(api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/api.h)
set(api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)

# backward api file
set(bw_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/backward_api_gen.py)
set(bw_api_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/backward.yaml)
set(legacy_bw_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_backward.yaml)
set(bw_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/backward/backward_api.h)
set(bw_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/backward_api.cc)
set(bw_api_header_file_tmp ${bw_api_header_file}.tmp)
set(bw_api_source_file_tmp ${bw_api_source_file}.tmp)

# dygraph(intermediate) api file
set(im_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/intermediate_api_gen.py)
set(dygraph_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/dygraph_api.h)
set(dygraph_api_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/dygraph_api.cc)
set(dygraph_api_header_file_tmp ${dygraph_api_header_file}.tmp)
set(dygraph_api_source_file_tmp ${dygraph_api_source_file}.tmp)

# sparse api file
set(sparse_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/sparse_api_gen.py)
set(sparse_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/sparse_ops.yaml)
set(sparse_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/sparse_api.h)
set(sparse_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/sparse_api.cc)
set(sparse_api_header_file_tmp ${sparse_api_header_file}.tmp)
set(sparse_api_source_file_tmp ${sparse_api_source_file}.tmp)

# sparse bw api file
set(sparse_bw_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/sparse_bw_api_gen.py)
set(sparse_bw_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/sparse_backward.yaml)
set(sparse_bw_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/backward/sparse_bw_api.h)
set(sparse_bw_api_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/sparse_bw_api.cc)
set(sparse_bw_api_header_file_tmp ${sparse_bw_api_header_file}.tmp)
set(sparse_bw_api_source_file_tmp ${sparse_bw_api_source_file}.tmp)

# strings api file
set(strings_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/strings_api_gen.py)
set(strings_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/strings_ops.yaml)
set(strings_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/strings_api.h)
set(strings_api_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/strings_api.cc)
set(strings_api_header_file_tmp ${strings_api_header_file}.tmp)
set(strings_api_source_file_tmp ${strings_api_source_file}.tmp)

# wrapped infermeta file
set(wrapped_infermeta_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/wrapped_infermeta_gen.py)
set(wrapped_infermeta_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.h)
set(wrapped_infermeta_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.cc)

# op extra info file
set(ops_extra_info_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/generator/ops_extra_info_gen.py)
set(op_compat_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)
set(ops_extra_info_file
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/ops_extra_info.cc)

if(NOT PYTHONINTERP_FOUND)
  find_package(PythonInterp REQUIRED)
endif()

# install extra dependencies
if(${PYTHON_VERSION_STRING} VERSION_LESS "3.6.2")
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install -U pyyaml
                          typing-extensions>=4.1.1 jinja2==2.11.3)
else()
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install -U pyyaml jinja2
                          typing-extensions)
endif()

# parse apis
set(parsed_api_dir ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/parsed_apis)
set(generated_op_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_op.cc)
set(generated_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/compat/generated_sig.cc)
message(
  "parse api yamls:
- ${api_yaml_file}
- ${legacy_api_yaml_file}
- ${bw_api_yaml_file}
- ${legacy_bw_api_yaml_file}")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_api_dir}
  COMMAND ${PYTHON_EXECUTABLE} generator/parse_api.py --api_yaml_path ./ops.yaml
          --output_path ./parsed_apis/api.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} generator/parse_api.py --api_yaml_path
          ./legacy_ops.yaml --output_path ./parsed_apis/legacy_api.parsed.yaml
  COMMAND
    ${PYTHON_EXECUTABLE} generator/parse_api.py --api_yaml_path ./backward.yaml
    --output_path ./parsed_apis/backward_api.parsed.yaml --backward
  COMMAND
    ${PYTHON_EXECUTABLE} generator/parse_api.py --api_yaml_path
    ./legacy_backward.yaml --output_path
    ./parsed_apis/legacy_backward_api.parsed.yaml --backward RESULTS_VARIABLE
    _results)
foreach(_result in ${_results})
  if(${_result})
    message(FATAL_ERROR "api yaml parsing failed, exiting.")
  endif()
endforeach()

# validation of api yamls
message("validate api yaml:
- ${parsed_api_dir}/api.parsed.yaml
- ${parsed_api_dir}/backward_api.parsed.yaml")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml
  COMMAND
    ${PYTHON_EXECUTABLE} generator/cross_validate.py --forward_yaml_paths
    ./parsed_apis/api.parsed.yaml ./parsed_apis/legacy_api.parsed.yaml
    --backward_yaml_paths ./parsed_apis/backward_api.parsed.yaml
    ./parsed_apis/legacy_backward_api.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "api validation failed, exiting.")
endif()

# code generation for op, op makers, and argument mapping functions
message(
  "create or remove auto-geneated operators: ${generated_op_path}.tmp
create or remove auto-geneated argument mappings: ${generated_argument_mapping_path}.tmp"
)
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml
  COMMAND
    ${PYTHON_EXECUTABLE} generator/generate_op.py --api_yaml_path
    ./parsed_apis/api.parsed.yaml --backward_api_yaml_path
    ./parsed_apis/backward_api.parsed.yaml --api_version_yaml_path
    op_version.yaml --op_compat_yaml_path op_compat.yaml --output_op_path
    "${generated_op_path}.tmp" --output_arg_map_path
    "${generated_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

if(EXISTS "${generated_op_path}.tmp" AND EXISTS "${generated_op_path}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                          "${generated_op_path}.tmp" "${generated_op_path}")
  message("copy if different ${generated_op_path}.tmp ${generated_op_path}")
elseif(EXISTS "${generated_op_path}.tmp")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${generated_op_path}.tmp"
                          "${generated_op_path}")
  message("copy ${generated_op_path}.tmp ${generated_op_path}")
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f "${generated_op_path}")
  message("remove ${generated_op_path}")
endif()

if(EXISTS "${generated_argument_mapping_path}.tmp"
   AND EXISTS "${generated_argument_mapping_path}")
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      "${generated_argument_mapping_path}.tmp"
      "${generated_argument_mapping_path}")
  message(
    "copy if different ${generated_argument_mapping_path}.tmp ${generated_argument_mapping_path}"
  )
elseif(EXISTS "${generated_argument_mapping_path}.tmp")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy "${generated_argument_mapping_path}.tmp"
            "${generated_argument_mapping_path}")
  message(
    "copy ${generated_argument_mapping_path}.tmp ${generated_argument_mapping_path}"
  )
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f
                          "${generated_argument_mapping_path}")
  message("remove ${generated_argument_mapping_path}")
endif()

# generate ops extra info
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${ops_extra_info_gen_file} --op_compat_yaml_path
          ${op_compat_yaml_file} --ops_extra_info_path ${ops_extra_info_file})
message("generate ${ops_extra_info_file}")

# generate forward api
add_custom_command(
  OUTPUT ${api_header_file} ${api_source_file}
  COMMAND ${PYTHON_EXECUTABLE} -m pip install pyyaml
  COMMAND
    ${PYTHON_EXECUTABLE} ${api_gen_file} --api_yaml_path ${api_yaml_file}
    ${legacy_api_yaml_file} --api_header_path ${api_header_file_tmp}
    --api_header_path ${api_header_file_tmp} --api_source_path
    ${api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_header_file_tmp}
          ${api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_source_file_tmp}
          ${api_source_file}
  COMMENT "copy_if_different ${api_header_file} ${api_source_file}"
  DEPENDS ${api_yaml_file} ${legacy_api_yaml_file} ${api_gen_file}
          ${api_gen_base}
  VERBATIM)

# generate backward api
add_custom_command(
  OUTPUT ${bw_api_header_file} ${bw_api_source_file} ${bw_api_header_file_tmp}
         ${bw_api_source_file_tmp}
  COMMAND
    ${PYTHON_EXECUTABLE} ${bw_api_gen_file} --backward_yaml_path
    ${bw_api_yaml_file} ${legacy_bw_api_yaml_file} --backward_header_path
    ${bw_api_header_file_tmp} --backward_source_path ${bw_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${bw_api_header_file_tmp}
          ${bw_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${bw_api_source_file_tmp}
          ${bw_api_source_file}
  COMMENT "copy_if_different ${bw_api_header_file} ${bw_api_source_file}"
  DEPENDS ${bw_api_yaml_file} ${bw_api_gen_file} ${api_gen_base}
          ${legacy_bw_api_yaml_file}
  VERBATIM)

# generate sparse api
add_custom_command(
  OUTPUT ${sparse_api_header_file} ${sparse_api_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${sparse_api_gen_file} --api_yaml_path
    ${sparse_api_yaml_file} --api_header_path ${sparse_api_header_file_tmp}
    --api_source_path ${sparse_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_api_header_file_tmp}
          ${sparse_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_api_source_file_tmp}
          ${sparse_api_source_file}
  COMMENT
    "copy_if_different ${sparse_api_header_file} ${sparse_sparse_api_source_file}"
  DEPENDS ${sparse_api_yaml_file} ${sparse_api_gen_file} ${api_gen_base}
          ${api_gen_file}
  VERBATIM)

# generate backward sparse api
add_custom_command(
  OUTPUT ${sparse_bw_api_header_file} ${sparse_bw_api_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${sparse_bw_api_gen_file} --api_yaml_path
    ${sparse_bw_api_yaml_file} --api_header_path
    ${sparse_bw_api_header_file_tmp} --api_source_path
    ${sparse_bw_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_bw_api_header_file_tmp}
          ${sparse_bw_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_bw_api_source_file_tmp}
          ${sparse_bw_api_source_file}
  COMMENT
    "copy_if_different ${sparse_bw_api_header_file} ${sparse_bw_sparse_api_source_file}"
  DEPENDS ${sparse_bw_api_yaml_file} ${sparse_bw_api_gen_file} ${api_gen_base}
          ${api_gen_file} ${sparse_api_gen_file} ${bw_api_gen_file}
  VERBATIM)

# generate strings api
add_custom_command(
  OUTPUT ${strings_api_header_file} ${strings_api_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${strings_api_gen_file} --api_yaml_path
    ${strings_api_yaml_file} --api_header_path ${strings_api_header_file_tmp}
    --api_source_path ${strings_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${strings_api_header_file_tmp}
          ${strings_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${strings_api_source_file_tmp}
          ${strings_api_source_file}
  COMMENT
    "copy_if_different ${strings_api_header_file} ${strings_strings_api_source_file}"
  DEPENDS ${strings_api_yaml_file} ${strings_api_gen_file} ${api_gen_base}
          ${api_gen_file}
  VERBATIM)

# generate dygraph(intermediate) api
add_custom_command(
  OUTPUT ${dygraph_api_header_file} ${dygraph_api_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${im_api_gen_file} --api_yaml_path ${api_yaml_file}
    ${legacy_api_yaml_file} --sparse_api_yaml_path ${sparse_api_yaml_file}
    --dygraph_api_header_path ${dygraph_api_header_file_tmp}
    --dygraph_api_source_path ${dygraph_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dygraph_api_header_file_tmp}
          ${dygraph_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dygraph_api_source_file_tmp}
          ${dygraph_api_source_file}
  DEPENDS ${api_yaml_file} ${legacy_api_yaml_file} ${sparse_api_yaml_file}
          ${im_api_gen_file} ${api_gen_base} ${api_gen_file}
  VERBATIM)

# generate wrapped infermeta
add_custom_command(
  OUTPUT ${wrapped_infermeta_header_file} ${wrapped_infermeta_source_file}
  COMMAND
    ${PYTHON_EXECUTABLE} ${wrapped_infermeta_gen_file} --api_yaml_path
    ${api_yaml_file} ${legacy_api_yaml_file} --wrapped_infermeta_header_path
    ${wrapped_infermeta_header_file} --wrapped_infermeta_source_path
    ${wrapped_infermeta_source_file}
  DEPENDS ${api_yaml_file} ${legacy_api_yaml_file} ${wrapped_infermeta_gen_file}
          ${api_gen_base}
  VERBATIM)

cc_library(
  op_meta_info
  SRCS op_meta_info.cc
  DEPS phi_tensor_raw)
cc_library(
  wrapped_infermeta
  SRCS ${wrapped_infermeta_source_file}
  DEPS phi)
cc_library(
  context_pool
  SRCS context_pool.cc
  DEPS phi_backends phi_enforce place init)

cc_library(
  kernel_dispatch
  SRCS kernel_dispatch.cc
  DEPS phi_tensor_raw phi_backends kernel_factory context_pool)
cc_library(
  api_gen_utils
  SRCS api_gen_utils.cc
  DEPS phi_tensor_raw selected_rows sparse_csr_tensor sparse_coo_tensor
       infermeta_utils)
cc_library(
  phi_data_transform
  SRCS data_transform.cc
  DEPS phi_tensor_raw phi tensor)
cc_library(
  api_custom_impl
  SRCS api_custom_impl.cc
  DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils backward_infermeta
       phi_data_transform)

cc_library(
  phi_function_api
  SRCS ${api_source_file}
  DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils phi_data_transform
       api_custom_impl)
cc_library(
  phi_bw_function_api
  SRCS ${bw_api_source_file}
  DEPS phi_tensor_raw
       phi
       kernel_dispatch
       api_gen_utils
       backward_infermeta
       phi_data_transform
       phi_function_api
       api_custom_impl
       global_utils)
cc_library(
  sparse_api
  SRCS ${sparse_api_source_file}
  DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils)
cc_library(
  sparse_bw_api
  SRCS ${sparse_bw_api_source_file}
  DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils sparse_api)
cc_library(
  phi_dygraph_api
  SRCS ${dygraph_api_source_file}
  DEPS phi_tensor_raw
       phi
       kernel_dispatch
       api_gen_utils
       phi_data_transform
       phi_function_api
       sparse_api)
cc_library(
  strings_api
  SRCS ${strings_api_source_file}
  DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils)
cc_library(
  phi_tensor
  SRCS tensor_method.cc
  DEPS phi_tensor_raw
       phi_function_api
       api_gen_utils
       kernel_dispatch
       infermeta
       sparse_api
       strings_api)
cc_library(
  tensor_copy
  SRCS tensor_copy.cc
  DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils)
cc_library(
  api_scalar
  SRCS scalar.cc
  DEPS tensor_copy)
cc_library(
  api_int_array
  SRCS int_array.cc
  DEPS tensor_copy)
