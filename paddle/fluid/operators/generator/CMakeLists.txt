# phi auto cmake utils
include(phi)

# set yaml file path
set(op_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/ops.yaml)
set(legacy_op_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_ops.yaml)
set(bw_op_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/backward.yaml)
set(static_op_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/static_ops.yaml)
set(fused_op_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/fused_ops.yaml)
set(legacy_bw_op_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_backward.yaml)
set(sparse_op_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/sparse_ops.yaml)
set(sparse_bw_op_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/sparse_backward.yaml)
set(static_bw_op_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/static_backward.yaml)
set(fused_bw_op_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/fused_backward.yaml)

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

# parse ops
set(parsed_op_dir
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator/parsed_ops)
set(generated_op_path_1
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_op1.cc)
set(generated_op_path_2
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_op2.cc)
set(generated_op_path_3
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_op3.cc)
set(generated_op_path_4
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_op4.cc)
set(generated_static_op_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_static_op.cc)
set(generated_fused_op_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_fused_op.cc)
set(generated_sparse_ops_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_sparse_op.cc)
set(generated_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/compat/generated_sig.cc)
set(generated_fused_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/compat/generated_fused_sig.cc)
set(generated_static_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/compat/generated_static_sig.cc)
set(generated_sparse_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/compat/generated_sparse_sig.cc)

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir} RESULTS_VARIABLE
          _result)
if(${_result})
  message(FATAL_ERROR "Failed to create folder for parsed op yamls, exiting.")
endif()

message(
  "parse op yamls:
- ${op_yaml_file}
- ${legacy_op_yaml_file}
- ${bw_op_yaml_file}
- ${legacy_bw_op_yaml_file}
- ${fused_op_yaml_file}
- ${static_op_yaml_file}")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${op_yaml_file}
          --output_path ./parsed_ops/ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${legacy_op_yaml_file}
          --output_path ./parsed_ops/legacy_ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${bw_op_yaml_file}
          --output_path ./parsed_ops/backward_ops.parsed.yaml --backward
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${legacy_bw_op_yaml_file}
    --output_path ./parsed_ops/legacy_backward_ops.parsed.yaml --backward
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${static_op_yaml_file}
          --output_path ./parsed_ops/static_ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${fused_op_yaml_file}
          --output_path ./parsed_ops/fused_ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${sparse_op_yaml_file}
          --output_path ./parsed_ops/sparse_ops.parsed.yaml
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${sparse_bw_op_yaml_file}
    --output_path ./parsed_ops/sparse_backward.parsed.yaml --backward
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${static_bw_op_yaml_file}
    --output_path ./parsed_ops/static_backward.parsed.yaml --backward
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${fused_bw_op_yaml_file}
    --output_path ./parsed_ops/fused_backward.parsed.yaml --backward
    RESULTS_VARIABLE _results)
foreach(_result in ${_results})
  if(${_result})
    message(FATAL_ERROR "op yaml parsing failed, exiting.")
  endif()
endforeach()

# validation of op yamls
message("validate op yaml:
- ${parsed_op_dir}/ops.parsed.yaml
- ${parsed_op_dir}/backward_ops.parsed.yaml")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ./parsed_ops/ops.parsed.yaml ./parsed_ops/legacy_ops.parsed.yaml
    --backward_yaml_paths ./parsed_ops/backward_ops.parsed.yaml
    ./parsed_ops/legacy_backward_ops.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "ops validation failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ./parsed_ops/static_ops.parsed.yaml --backward_yaml_paths
    ./parsed_ops/static_backward.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "static ops validation failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ./parsed_ops/fused_ops.parsed.yaml --backward_yaml_paths
    ./parsed_ops/fused_backward.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "fused ops validation failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ./parsed_ops/sparse_ops.parsed.yaml --backward_yaml_paths
    ./parsed_ops/sparse_backward.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "sparse ops validation failed, exiting.")
endif()

# code generation for op, op makers, and argument mapping functions
message(
  "create or remove auto-geneated operators: generated_op(1-4).cc.tmp
create or remove auto-geneated argument mappings: ${generated_argument_mapping_path}.tmp"
)
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_op.py --ops_yaml_path
    ./parsed_ops/ops.parsed.yaml --backward_yaml_path
    ./parsed_ops/backward_ops.parsed.yaml --op_version_yaml_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_version.yaml
    --op_compat_yaml_path ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml
    --output_op_path "${generated_op_path_1}.tmp" "${generated_op_path_2}.tmp"
    "${generated_op_path_3}.tmp" "${generated_op_path_4}.tmp"
    --output_arg_map_path "${generated_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_op.py --ops_yaml_path
    ./parsed_ops/static_ops.parsed.yaml --backward_yaml_path
    ./parsed_ops/static_backward.parsed.yaml --op_version_yaml_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_version.yaml
    --op_compat_yaml_path ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml
    --output_op_path "${generated_static_op_path}.tmp" --output_arg_map_path
    "${generated_static_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_op.py --ops_yaml_path
    ./parsed_ops/fused_ops.parsed.yaml --backward_yaml_path
    ./parsed_ops/fused_backward.parsed.yaml --op_version_yaml_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_version.yaml
    --op_compat_yaml_path ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml
    --output_op_path "${generated_fused_op_path}.tmp" --output_arg_map_path
    "${generated_fused_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_sparse_op.py --ops_yaml_path
    ./parsed_ops/sparse_ops.parsed.yaml --backward_ops_yaml_path
    ./parsed_ops/sparse_backward.parsed.yaml --output_op_path
    "${generated_sparse_ops_path}.tmp" --output_arg_map_path
    "${generated_sparse_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "sparse operator codegen failed, exiting.")
endif()

set(generated_static_files
    "${generated_op_path_1}"
    "${generated_op_path_2}"
    "${generated_op_path_3}"
    "${generated_op_path_4}"
    "${generated_static_op_path}"
    "${generated_fused_op_path}"
    "${generated_sparse_ops_path}"
    "${generated_argument_mapping_path}"
    "${generated_static_argument_mapping_path}"
    "${generated_sparse_argument_mapping_path}"
    "${generated_fused_argument_mapping_path}")

foreach(generated_static_file ${generated_static_files})
  if(EXISTS "${generated_static_file}.tmp" AND EXISTS
                                               "${generated_static_file}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${generated_static_file}.tmp" "${generated_static_file}")
    message(
      "copy if different ${generated_static_file}.tmp ${generated_static_file}")
  elseif(EXISTS "${generated_static_file}.tmp")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy "${generated_static_file}.tmp"
              "${generated_static_file}")
    message("copy ${generated_static_file}.tmp ${generated_static_file}")
  else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f
                            "${generated_static_file}")
    message("remove ${generated_static_file}")
  endif()
endforeach()

# Note(zyfncg): The generated file generated_op.cc has been deleted,
# so we need to clear the generated_op.cc and generated_op.cc.tmp cached in develop environment.
set(old_generated_op_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_op.cc)
if(EXISTS "${old_generated_op_path}" OR EXISTS "${old_generated_op_path}.tmp")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E remove -f "${old_generated_op_path}"
            "${old_generated_op_path}.tmp")
endif()

# op extra info file
set(ops_extra_info_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator/ops_extra_info_gen.py)
set(op_compat_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)
set(ops_extra_info_file
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/ops_extra_info.cc)

# generate ops extra info
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${ops_extra_info_gen_file} --op_compat_yaml_path
          ${op_compat_yaml_file} --ops_extra_info_path ${ops_extra_info_file})
message("generate ${ops_extra_info_file}")

set(op_utils_header
    ${PADDLE_BINARY_DIR}/paddle/phi/ops/compat/signatures.h.tmp
    CACHE INTERNAL "op_args_fns.cc file")
set(op_utils_header_final
    ${PADDLE_BINARY_DIR}/paddle/phi/ops/compat/signatures.h)
file(
  WRITE ${op_utils_header}
  "// Generated by the paddle/fluid/operators/generator/CMakeLists.txt.  DO NOT EDIT!\n\n"
)
file(APPEND ${op_utils_header}
     "#include \"paddle/phi/core/compat/op_utils.h\"\n\n")

# Automatically generate the registration code of all arg map functions
# and compile the corresponding target to avoid frequent code conflicts
# when writing to same file
register_op_utils(op_compat_infos DEPS op_utils)

copy_if_different(${op_utils_header} ${op_utils_header_final})

# add special GetExpectedKernelType
cc_library(
  get_expected_kernel_func
  SRCS get_expected_kernel_func.cc
  DEPS operator)
