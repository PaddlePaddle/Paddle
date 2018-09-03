# windows treat symbolic file as a real file, which is different with unix
# We create a hidden file and compile it instead of origin source file.
function(windows_symbolic TARGET)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(windows_symbolic "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  foreach(src ${windows_symbolic_SRCS})
  get_filename_component(src ${src} NAME_WE)
  if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src}.cc OR NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src}.cu)
      message(FATAL " ${src}.cc and ${src}.cu must exsits, and ${src}.cu must be symbolic file.")
  endif()
  add_custom_command(OUTPUT .${src}.cu 
          COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_SOURCE_DIR}/.${src}.cu
          COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${src}.cc" "${CMAKE_CURRENT_SOURCE_DIR}/.${src}.cu"
          COMMENT "create hidden file of ${src}.cu")
  add_custom_target(${TARGET} ALL DEPENDS .${src}.cu)  
  endforeach()
endfunction()

add_subdirectory(ir)
if (NOT WIN32)
add_subdirectory(details)
endif (NOT WIN32)
# ddim lib
proto_library(framework_proto SRCS framework.proto)

cc_library(ddim SRCS ddim.cc DEPS eigen3 boost)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)
cc_library(data_type SRCS data_type.cc DEPS framework_proto ddim device_context)
cc_test(data_type_test SRCS data_type_test.cc DEPS data_type place tensor)
if(WITH_GPU)
  if (WIN32)
    windows_symbolic(tensor_util SRCS tensor_util.cu)
    nv_library(tensor SRCS tensor.cc .tensor_util.cu DEPS place memory data_type device_context)
    add_dependencies(tensor tensor_util)
  else()
    nv_library(tensor SRCS tensor.cc tensor_util.cu DEPS place memory data_type device_context)
  endif(WIN32)
else()
  cc_library(tensor SRCS tensor.cc tensor_util.cc DEPS place memory data_type device_context)
endif()

cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)
if(WITH_GPU)
  nv_test(tensor_util_test SRCS tensor_util_test.cc tensor_util_test.cu DEPS tensor)
else()
  cc_test(tensor_util_test SRCS tensor_util_test.cc DEPS tensor)
endif()

cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

if(WITH_GPU)
  nv_test(mixed_vector_test SRCS mixed_vector_test.cc mixed_vector_test.cu DEPS place memory device_context tensor)
else()
  cc_test(mixed_vector_test SRCS mixed_vector_test.cc DEPS place memory device_context tensor)
endif()
if (NOT WIN32)
cc_library(lod_tensor SRCS lod_tensor.cc DEPS ddim place tensor framework_proto recordio)
else()
cc_library(lod_tensor SRCS lod_tensor.cc DEPS ddim place tensor framework_proto)
endif (NOT WIN32)

cc_test(lod_tensor_test SRCS lod_tensor_test.cc DEPS lod_tensor memory)
nv_test(lod_tensor_gpu_test SRCS lod_tensor_test.cu DEPS lod_tensor)

cc_library(reader SRCS reader.cc DEPS lod_tensor ddim)
cc_test(reader_test SRCS reader_test.cc DEPS reader)

cc_test(variable_test SRCS variable_test.cc)

cc_library(threadpool SRCS threadpool.cc DEPS enforce)
cc_test(threadpool_test SRCS threadpool_test.cc DEPS threadpool)

cc_library(scope SRCS scope.cc DEPS glog threadpool)
cc_test(scope_test SRCS scope_test.cc DEPS scope)

cc_library(data_device_transform SRCS data_device_transform.cc DEPS tensor)
nv_test(data_device_transform_test SRCS data_device_transform_test.cu
        DEPS operator op_registry device_context math_function)

if(WITH_GPU)
  if (WIN32)
      windows_symbolic(hidden_file SRCS data_type_transform.cu)
      nv_library(data_type_transform SRCS .data_type_transform.cu DEPS tensor)
      add_dependencies(data_type_transform hidden_file)
  else()
      nv_library(data_type_transform SRCS data_type_transform.cu DEPS tensor)
  endif(WIN32)
  nv_test(data_type_transform_test SRCS data_type_transform_test.cc data_type_transform_test.cu DEPS data_type_transform)
else()
  cc_library(data_type_transform SRCS data_type_transform.cc DEPS tensor)
  cc_test(data_type_transform_test SRCS data_type_transform_test.cc DEPS data_type_transform)
endif()

cc_library(data_layout_transform SRCS data_layout_transform.cc DEPS tensor math_function)
cc_test(data_layout_transform_test SRCS data_layout_transform_test.cc DEPS data_layout_transform)

cc_library(data_transform SRCS data_transform.cc DEPS math_function tensor
        framework_proto selected_rows data_device_transform data_type_transform data_layout_transform)

cc_library(attribute SRCS attribute.cc DEPS framework_proto boost)
cc_test(program_desc_test SRCS program_desc_test.cc DEPS proto_desc
device_context)
cc_library(op_proto_maker SRCS op_proto_maker.cc DEPS framework_proto attribute glog)
cc_test(op_proto_maker_test SRCS op_proto_maker_test.cc DEPS op_proto_maker)
cc_library(op_info SRCS op_info.cc DEPS attribute framework_proto)
cc_library(shape_inference SRCS shape_inference.cc DEPS ddim attribute device_context)

if (NOT WIN32)
cc_library(operator SRCS operator.cc DEPS op_info device_context tensor scope glog
    shape_inference data_transform lod_tensor profiler)
else()
cc_library(operator SRCS operator.cc DEPS op_info device_context tensor scope glog
    shape_inference data_transform lod_tensor)
endif(NOT WIN32)

cc_test(operator_test SRCS operator_test.cc DEPS operator op_registry device_context)
cc_library(proto_desc SRCS var_desc.cc op_desc.cc block_desc.cc program_desc.cc DEPS shape_inference op_info operator glog)

cc_library(op_registry SRCS op_registry.cc DEPS op_proto_maker op_info operator glog proto_desc)
nv_test(op_registry_test SRCS op_registry_test.cc DEPS op_registry)

if (NOT WIN32)
py_proto_compile(framework_py_proto SRCS framework.proto)
# Generate an empty __init__.py to make framework_py_proto as a valid python module.
add_custom_target(framework_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)
add_dependencies(framework_py_proto framework_py_proto_init)
add_custom_command(TARGET framework_py_proto POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto
    COMMAND cp *.py ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/
    COMMENT "Copy generated python proto into directory paddle/fluid/proto."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif(NOT WIN32)

cc_library(lod_rank_table SRCS lod_rank_table.cc DEPS lod_tensor)

cc_library(feed_fetch_method SRCS feed_fetch_method.cc DEPS lod_tensor scope glog)

if(WITH_DISTRIBUTE)
  cc_library(executor SRCS executor.cc DEPS op_registry device_context scope framework_proto glog lod_rank_table feed_fetch_method sendrecvop_grpc cares grpc++_unsecure grpc_unsecure gpr graph_to_program_pass)
  set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
  set_source_files_properties(executor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
else()
  cc_library(executor SRCS executor.cc DEPS op_registry device_context scope framework_proto glog lod_rank_table feed_fetch_method graph_to_program_pass)
endif()

if (NOT WIN32)
cc_library(parallel_executor SRCS parallel_executor.cc DEPS
        threaded_ssa_graph_executor scope_buffered_ssa_graph_executor
        graph graph_viz_pass multi_devices_graph_pass
        multi_devices_graph_print_pass multi_devices_graph_check_pass
        fast_threaded_ssa_graph_executor)
endif() # NOT WIN32

cc_library(prune SRCS prune.cc DEPS framework_proto)
cc_test(prune_test SRCS prune_test.cc DEPS op_info prune recurrent_op device_context)
cc_test(var_type_inference_test SRCS var_type_inference_test.cc DEPS op_registry
        proto_desc)
cc_library(selected_rows SRCS selected_rows.cc DEPS tensor)
cc_test(selected_rows_test SRCS selected_rows_test.cc DEPS selected_rows)

cc_test(op_kernel_type_test SRCS op_kernel_type_test.cc DEPS place device_context framework_proto)
cc_test(cow_ptr_tests SRCS details/cow_ptr_test.cc)

# cc_test(channel_test SRCS channel_test.cc)
cc_test(tuple_test SRCS tuple_test.cc )

if (NOT WIN32)
cc_test(rw_lock_test SRCS rw_lock_test.cc)
endif (NOT WIN32)

# disable test temporarily.
# TODO https://github.com/PaddlePaddle/Paddle/issues/11971
# cc_test(concurrency_test SRCS concurrency_test.cc DEPS go_op channel_close_op channel_create_op
#         channel_send_op channel_recv_op sum_op select_op elementwise_add_op compare_op
#         conditional_block_op while_op assign_op print_op executor proto_desc)
