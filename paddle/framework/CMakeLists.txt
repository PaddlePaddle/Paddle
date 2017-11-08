# ddim lib
proto_library(framework_proto SRCS framework.proto)

cc_library(ddim SRCS ddim.cc DEPS eigen3)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

cc_library(tensor SRCS tensor.cc DEPS ddim place paddle_memory device_context)
cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)
cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

cc_library(lod_tensor SRCS lod_tensor.cc DEPS ddim place tensor framework_proto)
cc_test(lod_tensor_test SRCS lod_tensor_test.cc DEPS lod_tensor paddle_memory)
nv_test(lod_tensor_gpu_test SRCS lod_tensor_test.cu DEPS lod_tensor)

cc_test(variable_test SRCS variable_test.cc)

cc_library(scope SRCS scope.cc DEPS glog)
cc_test(scope_test SRCS scope_test.cc DEPS scope)


cc_library(attribute SRCS attribute.cc DEPS framework_proto)
cc_test(program_desc_test SRCS program_desc_test.cc DEPS proto_desc
device_context)
cc_library(op_proto_maker SRCS op_proto_maker.cc DEPS framework_proto attribute)
cc_test(op_proto_maker_test SRCS op_proto_maker_test.cc DEPS op_proto_maker)
cc_library(op_info SRCS op_info.cc DEPS attribute framework_proto)
cc_library(shape_inference SRCS shape_inference.cc DEPS ddim attribute)
cc_library(operator SRCS operator.cc DEPS op_info device_context tensor scope glog shape_inference)
cc_test(operator_test SRCS operator_test.cc DEPS operator op_registry)
cc_library(proto_desc SRCS var_desc.cc op_desc.cc block_desc.cc program_desc.cc DEPS shape_inference op_info operator glog)

cc_library(op_registry SRCS op_registry.cc DEPS op_proto_maker op_info operator glog proto_desc)
cc_test(op_registry_test SRCS op_registry_test.cc DEPS op_registry)

py_proto_compile(framework_py_proto SRCS framework.proto)
# Generate an empty __init__.py to make framework_py_proto as a valid python module.
add_custom_target(framework_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)
add_dependencies(framework_py_proto framework_py_proto_init)
add_custom_command(TARGET framework_py_proto POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/proto
    COMMAND cp *.py ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/proto/
    COMMENT "Copy generated python proto into directory paddle/v2/framework/proto."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

cc_library(backward SRCS backward.cc DEPS net_op)
cc_test(backward_test SRCS backward_test.cc DEPS backward recurrent_op device_context fill_constant_op)
cc_library(lod_rank_table SRCS lod_rank_table.cc DEPS lod_tensor)

cc_library(executor SRCS executor.cc DEPS op_registry device_context scope framework_proto backward glog lod_rank_table)

cc_library(prune SRCS prune.cc DEPS framework_proto)
cc_test(prune_test SRCS prune_test.cc DEPS op_info prune recurrent_op device_context)

cc_library(tensor_array SRCS tensor_array.cc DEPS lod_tensor)
cc_test(tensor_array_test SRCS tensor_array_test.cc DEPS tensor_array place)

cc_test(var_type_inference_test SRCS var_type_inference_test.cc DEPS op_registry
        proto_desc)
cc_library(selected_rows SRCS selected_rows.cc DEPS tensor)
cc_test(selected_rows_test SRCS selected_rows_test.cc DEPS selected_rows)
