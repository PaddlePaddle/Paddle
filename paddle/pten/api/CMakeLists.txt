add_subdirectory(lib)

set(api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api.yaml)

set(api_header_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/include/api.h)
set(api_source_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/lib/api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)

add_custom_command(
  OUTPUT ${api_header_file} ${api_source_file}
  COMMAND python3.7 ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api_gen.py 
                 --api_yaml_path ${api_yaml_file}
                 --api_header_path ${api_header_file_tmp}
                 --api_source_path ${api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_header_file_tmp} ${api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_source_file_tmp} ${api_source_file}
  COMMENT "copy_if_different ${api_header_file} ${api_source_file}"
  DEPENDS ${api_yaml_file}
  VERBATIM)

cc_library(pten_function_api SRCS ${api_source_file} DEPS pten)
cc_library(pten_api SRCS all.cc DEPS linalg_api math_api creation_api manipulation_api pten_function_api)
