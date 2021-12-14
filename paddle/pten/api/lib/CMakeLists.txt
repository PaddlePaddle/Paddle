add_subdirectory(utils)

cc_library(ext_compat_utils SRCS ext_compat_utils.cc DEPS place)

if (WITH_GPU)
  nv_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils enforce)
elseif (WITH_ROCM)
  hip_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils enforce)
else()
  cc_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils enforce)
endif()

cc_library(kernel_dispatch SRCS kernel_dispatch.cc DEPS pten_tensor device_context kernel_factory)

cc_library(op_meta_info SRCS op_meta_info.cc DEPS pten_tensor)

set(api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api_gen.py)
set(api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api.yaml)

set(api_header_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/include/api.h)
set(api_source_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/lib/api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)

if (NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp REQUIRED)
endif()

add_custom_command(
  OUTPUT ${api_header_file} ${api_source_file}
  COMMAND ${PYTHON_EXECUTABLE} -m pip install pyyaml
  COMMAND ${PYTHON_EXECUTABLE} ${api_gen_file} 
                 --api_yaml_path ${api_yaml_file}
                 --api_header_path ${api_header_file_tmp}
                 --api_source_path ${api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_header_file_tmp} ${api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${api_source_file_tmp} ${api_source_file}
  COMMENT "copy_if_different ${api_header_file} ${api_source_file}"
  DEPENDS ${api_yaml_file} ${api_gen_file}
  VERBATIM)

cc_library(utils_api SRCS utils.cc DEPS pten_tensor pten kernel_dispatch)
cc_library(pten_function_api SRCS ${api_source_file} DEPS pten_tensor pten kernel_dispatch)
