add_subdirectory(utils)

cc_library(ext_compat_utils SRCS ext_compat_utils.cc DEPS place)

if (WITH_GPU)
  nv_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils pten_enforce manual_api pten_function_api)
elseif (WITH_ROCM)
  hip_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils pten_enforce manual_api pten_function_api)
else()
  cc_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils pten_enforce manual_api pten_function_api)
endif()

cc_library(kernel_dispatch SRCS kernel_dispatch.cc DEPS pten_tensor pten_context kernel_factory)

cc_library(op_meta_info SRCS op_meta_info.cc DEPS pten_tensor)
cc_library(op_kernel_info SRCS op_kernel_info.cc DEPS pten_tensor)


set(api_gen_base ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api_base.py)

# forward api file
set(api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api_gen.py)
set(api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api.yaml)
set(api_header_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/include/api.h)
set(api_source_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/lib/api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)

# backward api file
set(bw_api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/backward_api_gen.py)
set(bw_api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/backward.yaml)
set(bw_api_header_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/backward/backward_api.h)
set(bw_api_source_file ${CMAKE_SOURCE_DIR}/paddle/pten/api/lib/backward_api.cc)
set(bw_api_header_file_tmp ${bw_api_header_file}.tmp)
set(bw_api_source_file_tmp ${bw_api_source_file}.tmp)

# wrapped infermeta file
set(wrapped_infermeta_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/wrapped_infermeta_gen.py)
set(api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api.yaml)
set(wrapped_infermeta_header_file ${CMAKE_SOURCE_DIR}/paddle/pten/infermeta/generated.h)
set(wrapped_infermeta_source_file ${CMAKE_SOURCE_DIR}/paddle/pten/infermeta/generated.cc)

if (NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp REQUIRED)
endif()

# generate forward api
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
  DEPENDS ${api_yaml_file} ${api_gen_file} ${api_gen_base}
  VERBATIM)

# generate backward api
add_custom_command(
  OUTPUT ${bw_api_header_file} ${bw_api_source_file} ${bw_api_header_file_tmp} ${bw_api_source_file_tmp}
  COMMAND ${PYTHON_EXECUTABLE} ${bw_api_gen_file}
                 --backward_yaml_path ${bw_api_yaml_file}
                 --backward_header_path ${bw_api_header_file_tmp}
                 --backward_source_path ${bw_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${bw_api_header_file_tmp} ${bw_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${bw_api_source_file_tmp} ${bw_api_source_file}
  COMMENT "copy_if_different ${bw_api_header_file} ${bw_api_source_file}"
  DEPENDS ${bw_api_yaml_file} ${bw_api_gen_file} ${api_gen_base}
  VERBATIM)

# generate wrapped infermeta
add_custom_command(
  OUTPUT ${wrapped_infermeta_header_file} ${wrapped_infermeta_source_file}
  COMMAND ${PYTHON_EXECUTABLE} ${wrapped_infermeta_gen_file}
                 --api_yaml_path ${api_yaml_file}
                 --wrapped_infermeta_header_path ${wrapped_infermeta_header_file}
                 --wrapped_infermeta_source_path ${wrapped_infermeta_source_file}
  DEPENDS ${api_yaml_file} ${wrapped_infermeta_gen_file} ${api_gen_base}
  VERBATIM)

cc_library(pten_data_transform SRCS data_transform.cc DEPS pten_tensor transfer_layout_kernel cast_kernel data_device_transform)
cc_library(manual_api SRCS manual_api.cc DEPS pten_tensor pten kernel_dispatch)
cc_library(sparse_api SRCS sparse_api.cc DEPS pten_tensor pten kernel_dispatch pten_data_transform)
cc_library(pten_function_api SRCS ${api_source_file} DEPS pten_tensor pten kernel_dispatch pten_data_transform)
cc_library(pten_bw_function_api SRCS ${bw_api_source_file} DEPS pten_tensor pten kernel_dispatch backward_infermeta pten_data_transform pten_function_api)
cc_library(wrapped_infermeta SRCS ${wrapped_infermeta_source_file} DEPS pten)
