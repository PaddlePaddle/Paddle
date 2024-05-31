# phi auto cmake utils
include(operators)

# set yaml file path
set(ops_yaml_dir ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml)
set(op_yaml_file ${ops_yaml_dir}/ops.yaml)
set(legacy_op_yaml_file ${ops_yaml_dir}/inconsistent/dygraph_ops.yaml)
set(bw_op_yaml_file ${ops_yaml_dir}/backward.yaml)
set(static_op_yaml_file ${ops_yaml_dir}/legacy/static_ops.yaml)
set(fused_op_yaml_file ${ops_yaml_dir}/fused_ops.yaml)
set(legacy_bw_op_yaml_file ${ops_yaml_dir}/inconsistent/dygraph_backward.yaml)
set(sparse_op_yaml_file ${ops_yaml_dir}/sparse_ops.yaml)
set(sparse_bw_op_yaml_file ${ops_yaml_dir}/sparse_backward.yaml)
set(static_bw_op_yaml_file ${ops_yaml_dir}/legacy/static_backward.yaml)
set(fused_bw_op_yaml_file ${ops_yaml_dir}/fused_backward.yaml)

if(NOT PYTHONINTERP_FOUND)
  find_package(PythonInterp REQUIRED)
endif()

function(install_py_pyyaml)
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} "-c"
      "import re, yaml; print(re.compile('/__init__.py.*').sub('',yaml.__file__))"
    RESULT_VARIABLE _pyyaml_status
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT _pyyaml_status EQUAL 0)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install -U pyyaml)
  endif()

endfunction()

function(install_py_jinja2)
  # check install
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} "-c"
      "import re, jinja2; print(re.compile('/__init__.py.*').sub('',jinja2.__file__))"
    RESULT_VARIABLE _jinja2_status
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_jinja2_status EQUAL 0)
    # check version
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} "-c"
              "import jinja2; print(jinja2.__version__)"
      OUTPUT_VARIABLE _jinja2_version
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_jinja2_version)
      return()
    endif()
  endif()

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install -U jinja2)
endfunction()

install_py_pyyaml()
install_py_jinja2()

# parse ops
set(parsed_op_dir ${CMAKE_BINARY_DIR}/paddle/phi/ops/yaml)
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
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/ops_signature/generated_sig.cc)
set(generated_fused_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/ops_signature/generated_fused_sig.cc
)
set(generated_static_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/ops_signature/generated_static_sig.cc
)
set(generated_sparse_argument_mapping_path
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/ops_signature/generated_sparse_sig.cc
)
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
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir}/inconsistent
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir}/legacy
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${op_yaml_file}
          --output_path ${parsed_op_dir}/ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${legacy_op_yaml_file}
          --output_path ${parsed_op_dir}/inconsistent/dygraph_ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${bw_op_yaml_file}
          --output_path ${parsed_op_dir}/backward.parsed.yaml --backward
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${legacy_bw_op_yaml_file}
    --output_path ${parsed_op_dir}/inconsistent/dygraph_backward.parsed.yaml
    --backward
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${static_op_yaml_file}
          --output_path ${parsed_op_dir}/legacy/static_ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${fused_op_yaml_file}
          --output_path ${parsed_op_dir}/fused_ops.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${sparse_op_yaml_file}
          --output_path ${parsed_op_dir}/sparse_ops.parsed.yaml
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${sparse_bw_op_yaml_file}
    --output_path ${parsed_op_dir}/sparse_backward.parsed.yaml --backward
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${static_bw_op_yaml_file}
    --output_path ${parsed_op_dir}/legacy/static_backward.parsed.yaml --backward
  COMMAND
    ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${fused_bw_op_yaml_file}
    --output_path ${parsed_op_dir}/fused_backward.parsed.yaml --backward
    RESULTS_VARIABLE _results)
foreach(_result in ${_results})
  if(${_result})
    message(FATAL_ERROR "op yaml parsing failed, exiting.")
  endif()
endforeach()

# validation of op yamls
message("validate op yaml:
- ${parsed_op_dir}/ops.parsed.yaml
- ${parsed_op_dir}/backward.parsed.yaml")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ${parsed_op_dir}/ops.parsed.yaml
    ${parsed_op_dir}/inconsistent/dygraph_ops.parsed.yaml --backward_yaml_paths
    ${parsed_op_dir}/backward.parsed.yaml
    ${parsed_op_dir}/inconsistent/dygraph_backward.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "ops validation failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ${parsed_op_dir}/legacy/static_ops.parsed.yaml --backward_yaml_paths
    ${parsed_op_dir}/legacy/static_backward.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "static ops validation failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ${parsed_op_dir}/fused_ops.parsed.yaml --backward_yaml_paths
    ${parsed_op_dir}/fused_backward.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "fused ops validation failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ${parsed_op_dir}/sparse_ops.parsed.yaml --backward_yaml_paths
    ${parsed_op_dir}/sparse_backward.parsed.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "sparse ops validation failed, exiting.")
endif()

# code generation for op, op makers, and argument mapping functions
message(
  "create or remove auto-generated operators: generated_op(1-4).cc.tmp
create or remove auto-generated argument mappings: ${generated_argument_mapping_path}.tmp"
)
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_op.py --ops_yaml_path
    ${parsed_op_dir}/ops.parsed.yaml --backward_yaml_path
    ${parsed_op_dir}/backward.parsed.yaml --op_version_yaml_path
    ${ops_yaml_dir}/op_version.yaml --op_compat_yaml_path
    ${ops_yaml_dir}/op_compat.yaml --output_op_path "${generated_op_path_1}.tmp"
    "${generated_op_path_2}.tmp" "${generated_op_path_3}.tmp"
    "${generated_op_path_4}.tmp" --output_arg_map_path
    "${generated_argument_mapping_path}.tmp" --ops_exclude_yaml_path
    ${ops_yaml_dir}/legacy/ops_exclude.yaml --backward_exclude_yaml_path
    ${ops_yaml_dir}/legacy/backward_exclude.yaml
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_op.py --ops_yaml_path
    ${parsed_op_dir}/legacy/static_ops.parsed.yaml --backward_yaml_path
    ${parsed_op_dir}/legacy/static_backward.parsed.yaml --op_version_yaml_path
    ${ops_yaml_dir}/op_version.yaml --op_compat_yaml_path
    ${ops_yaml_dir}/op_compat.yaml --output_op_path
    "${generated_static_op_path}.tmp" --output_arg_map_path
    "${generated_static_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_op.py --ops_yaml_path
    ${parsed_op_dir}/fused_ops.parsed.yaml --backward_yaml_path
    ${parsed_op_dir}/fused_backward.parsed.yaml --op_version_yaml_path
    ${ops_yaml_dir}/op_version.yaml --op_compat_yaml_path
    ${ops_yaml_dir}/op_compat.yaml --output_op_path
    "${generated_fused_op_path}.tmp" --output_arg_map_path
    "${generated_fused_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_sparse_op.py --ops_yaml_path
    ${parsed_op_dir}/sparse_ops.parsed.yaml --backward_ops_yaml_path
    ${parsed_op_dir}/sparse_backward.parsed.yaml --output_op_path
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
set(op_compat_yaml_file ${ops_yaml_dir}/op_compat.yaml)
set(ops_extra_info_file
    ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/ops_extra_info.cc)

# generate ops extra info
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${ops_extra_info_gen_file} --op_compat_yaml_path
          ${op_compat_yaml_file} --ops_extra_info_path ${ops_extra_info_file})
message("generate ${ops_extra_info_file}")

set(op_utils_header
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/ops_signature/signatures.h.tmp
    CACHE INTERNAL "op_args_fns.cc file")
set(op_utils_header_final
    ${PADDLE_BINARY_DIR}/paddle/fluid/operators/ops_signature/signatures.h)
file(
  WRITE ${op_utils_header}
  "// Generated by the paddle/fluid/operators/generator/CMakeLists.txt.  DO NOT EDIT!\n\n"
)
file(APPEND ${op_utils_header}
     "#include \"paddle/phi/core/compat/op_utils.h\"\n\n")

# Automatically generate the registration code of all arg map functions
# and compile the corresponding target to avoid frequent code conflicts
# when writing to same file
register_op_utils(op_compat_infos DEPS phi common)

copy_if_different(${op_utils_header} ${op_utils_header_final})

# add special GetExpectedKernelType
cc_library(
  get_expected_kernel_func
  SRCS get_expected_kernel_func.cc
  DEPS operator)
