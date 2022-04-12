add_subdirectory(utils)
add_subdirectory(scripts)
add_subdirectory(testing)
set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests CACHE INTERNAL "python tests directory")

if (NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp REQUIRED)
endif()

# parse and validate apis
set(parsed_api_dir ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/parsed_apis)
set(generated_op_path ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generated_op.cc)
set(generated_argument_mapping_path ${CMAKE_SOURCE_DIR}/paddle/phi/ops/compat/generated_sig.cc)
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen
  COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_api_dir}
  COMMAND ${PYTHON_EXECUTABLE} parse_api.py 
      --api_yaml_path ./api.yaml 
      --output_path ./parsed_apis/api.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_api.py 
      --api_yaml_path ./new_api.yaml 
      --output_path ./parsed_apis/new_api.parsed.yaml
  COMMAND ${PYTHON_EXECUTABLE} parse_api.py
      --api_yaml_path ./backward.yaml
      --output_path ./parsed_apis/backward_api.parsed.yaml
      --backward
  COMMAND ${PYTHON_EXECUTABLE} parse_api.py
      --api_yaml_path ./new_backward.yaml
      --output_path ./parsed_apis/new_backward_api.parsed.yaml
      --backward
  RESULT_VARIABLE _result
)
if (_result) 
    message(FATAL_ERROR "api yaml parsing failed, exiting.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen
  COMMAND ${PYTHON_EXECUTABLE} cross_validate.py
      --forward_yaml_paths ./parsed_apis/api.parsed.yaml ./parsed_apis/new_api.parsed.yaml 
      --backward_yaml_paths ./parsed_apis/backward_api.parsed.yaml ./parsed_apis/new_backward_api.parsed.yaml
  RESULT_VARIABLE _result
)
if (_result) 
    message(FATAL_ERROR "api validation failed, exiting." )
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen
  COMMAND ${PYTHON_EXECUTABLE} generate_op.py
      --api_yaml_path ./parsed_apis/new_api.parsed.yaml
      --backward_api_yaml_path ./parsed_apis/new_backward_api.parsed.yaml
      --output_op_path "${generated_op_path}"
      --output_arg_map_path "${generated_argument_mapping_path}"
  RESULT_VARIABLE _result
)
if (_result)
    message(FATAL_ERROR "operator codegen failed, exiting." )
endif()

add_subdirectory(phi)
add_subdirectory(infrt)
add_subdirectory(fluid)
