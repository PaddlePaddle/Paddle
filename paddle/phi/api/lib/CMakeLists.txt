add_subdirectory(utils)

cc_library(ext_compat_utils SRCS ext_compat_utils.cc DEPS place)

if (WITH_GPU)
  nv_library(phi_tensor_raw SRCS tensor.cc DEPS tensor_base dense_tensor phi_api_utils ext_compat_utils phi_enforce)
elseif (WITH_ROCM)
  hip_library(phi_tensor_raw SRCS tensor.cc DEPS tensor_base dense_tensor phi_api_utils ext_compat_utils phi_enforce)
else()
  cc_library(phi_tensor_raw SRCS tensor.cc DEPS tensor_base dense_tensor phi_api_utils ext_compat_utils phi_enforce)
endif()

set(api_gen_base ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api_base.py)

# forward api file
set(api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api_gen.py)
set(api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api.yaml)
set(api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/api.h)
set(api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)

# backward api file
set(bw_api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/backward_api_gen.py)
set(bw_api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/backward.yaml)
set(bw_api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/backward/backward_api.h)
set(bw_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/backward_api.cc)
set(bw_api_header_file_tmp ${bw_api_header_file}.tmp)
set(bw_api_source_file_tmp ${bw_api_source_file}.tmp)

# dygraph(intermediate) api file
set(im_api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/intermediate_api_gen.py)
set(dygraph_api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/dygraph_api.h)
set(dygraph_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/dygraph_api.cc)
set(dygraph_api_header_file_tmp ${dygraph_api_header_file}.tmp)
set(dygraph_api_source_file_tmp ${dygraph_api_source_file}.tmp)

# sparse api file
set(sparse_api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/sparse_api_gen.py)
set(sparse_api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/sparse_api.yaml)
set(sparse_api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/sparse_api.h)
set(sparse_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/sparse_api.cc)
set(sparse_api_header_file_tmp ${sparse_api_header_file}.tmp)
set(sparse_api_source_file_tmp ${sparse_api_source_file}.tmp)

# sparse bw api file
set(sparse_bw_api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/sparse_bw_api_gen.py)
set(sparse_bw_api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/sparse_bw_api.yaml)
set(sparse_bw_api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/backward/sparse_bw_api.h)
set(sparse_bw_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/sparse_bw_api.cc)
set(sparse_bw_api_header_file_tmp ${sparse_bw_api_header_file}.tmp)
set(sparse_bw_api_source_file_tmp ${sparse_bw_api_source_file}.tmp)

# strings api file
set(strings_api_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/strings_api_gen.py)
set(strings_api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/strings_api.yaml)
set(strings_api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/strings_api.h)
set(strings_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/strings_api.cc)
set(strings_api_header_file_tmp ${strings_api_header_file}.tmp)
set(strings_api_source_file_tmp ${strings_api_source_file}.tmp)

# wrapped infermeta file
set(wrapped_infermeta_gen_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/wrapped_infermeta_gen.py)
set(api_yaml_file ${CMAKE_SOURCE_DIR}/python/paddle/utils/code_gen/api.yaml)
set(wrapped_infermeta_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.h)
set(wrapped_infermeta_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.cc)

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

# generate sparse api
add_custom_command(
  OUTPUT ${sparse_api_header_file} ${sparse_api_source_file}
  COMMAND ${PYTHON_EXECUTABLE} ${sparse_api_gen_file}
                 --api_yaml_path ${sparse_api_yaml_file}
                 --api_header_path ${sparse_api_header_file_tmp}
                 --api_source_path ${sparse_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_api_header_file_tmp} ${sparse_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_api_source_file_tmp} ${sparse_api_source_file}
  COMMENT "copy_if_different ${sparse_api_header_file} ${sparse_sparse_api_source_file}"
  DEPENDS ${sparse_api_yaml_file} ${sparse_api_gen_file} ${api_gen_base} ${api_gen_file}
  VERBATIM)

# generate backward sparse api
add_custom_command(
  OUTPUT ${sparse_bw_api_header_file} ${sparse_bw_api_source_file}
  COMMAND ${PYTHON_EXECUTABLE} ${sparse_bw_api_gen_file}
                 --api_yaml_path ${sparse_bw_api_yaml_file}
                 --api_header_path ${sparse_bw_api_header_file_tmp}
                 --api_source_path ${sparse_bw_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_bw_api_header_file_tmp} ${sparse_bw_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sparse_bw_api_source_file_tmp} ${sparse_bw_api_source_file}
  COMMENT "copy_if_different ${sparse_bw_api_header_file} ${sparse_bw_sparse_api_source_file}"
  DEPENDS ${sparse_bw_api_yaml_file} ${sparse_bw_api_gen_file} ${api_gen_base} ${api_gen_file} ${sparse_api_gen_file} ${bw_api_gen_file}
  VERBATIM)

# generate strings api
add_custom_command(
  OUTPUT ${strings_api_header_file} ${strings_api_source_file}
  COMMAND ${PYTHON_EXECUTABLE} ${strings_api_gen_file}
                 --api_yaml_path ${strings_api_yaml_file}
                 --api_header_path ${strings_api_header_file_tmp}
                 --api_source_path ${strings_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${strings_api_header_file_tmp} ${strings_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${strings_api_source_file_tmp} ${strings_api_source_file}
  COMMENT "copy_if_different ${strings_api_header_file} ${strings_strings_api_source_file}"
  DEPENDS ${strings_api_yaml_file} ${strings_api_gen_file} ${api_gen_base} ${api_gen_file}
  VERBATIM)

# generate dygraph(intermediate) api
add_custom_command(
  OUTPUT ${dygraph_api_header_file} ${dygraph_api_source_file}
  COMMAND ${PYTHON_EXECUTABLE} ${im_api_gen_file}
                 --api_yaml_path ${api_yaml_file}
                 --sparse_api_yaml_path ${sparse_api_yaml_file}
                 --dygraph_api_header_path ${dygraph_api_header_file_tmp}
                 --dygraph_api_source_path ${dygraph_api_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dygraph_api_header_file_tmp} ${dygraph_api_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dygraph_api_source_file_tmp} ${dygraph_api_source_file}
  DEPENDS ${api_yaml_file} ${sparse_api_yaml_file} ${im_api_gen_file} ${api_gen_base} ${api_gen_file}
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

cc_library(op_meta_info SRCS op_meta_info.cc DEPS phi_tensor_raw)
cc_library(wrapped_infermeta SRCS ${wrapped_infermeta_source_file} DEPS phi)
cc_library(context_pool SRCS context_pool.cc DEPS phi_context phi_enforce place)

cc_library(kernel_dispatch SRCS kernel_dispatch.cc DEPS phi_tensor_raw phi_context kernel_factory context_pool)
cc_library(api_gen_utils SRCS api_gen_utils.cc DEPS phi_tensor_raw selected_rows sparse_csr_tensor sparse_coo_tensor)
cc_library(phi_data_transform SRCS data_transform.cc DEPS phi_tensor_raw transfer_layout_kernel cast_kernel data_device_transform)
cc_library(api_custom_impl SRCS api_custom_impl.cc DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils phi_data_transform)
cc_library(sparse_api_custom_impl SRCS sparse_api_custom_impl.cc DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils phi_data_transform)

cc_library(phi_function_api SRCS ${api_source_file} DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils phi_data_transform api_custom_impl)
cc_library(phi_bw_function_api SRCS ${bw_api_source_file} DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils backward_infermeta phi_data_transform phi_function_api api_custom_impl global_utils)
cc_library(sparse_api SRCS ${sparse_api_source_file} DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils sparse_api_custom_impl)
cc_library(sparse_bw_api SRCS ${sparse_bw_api_source_file} DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils sparse_api sparse_api_custom_impl)
cc_library(phi_dygraph_api SRCS ${dygraph_api_source_file} DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils phi_data_transform phi_function_api sparse_api)
cc_library(strings_api SRCS ${strings_api_source_file} DEPS phi_tensor_raw phi kernel_dispatch api_gen_utils)
cc_library(phi_tensor SRCS tensor_method.cc DEPS phi_tensor_raw phi_function_api api_gen_utils kernel_dispatch infermeta sparse_api strings_api)
