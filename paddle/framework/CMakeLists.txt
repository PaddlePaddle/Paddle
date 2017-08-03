# ddim lib
cc_library(ddim SRCS ddim.cc DEPS eigen3)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

cc_library(tensor SRCS tensor.cc DEPS ddim place paddle_memory device_context)
cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)
cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

cc_test(variable_test SRCS variable_test.cc)

cc_library(scope SRCS scope.cc)
cc_test(scope_test SRCS scope_test.cc DEPS scope)

proto_library(attr_type SRCS attr_type.proto)
proto_library(op_proto SRCS op_proto.proto DEPS attr_type)
proto_library(op_desc SRCS op_desc.proto DEPS attr_type)
cc_test(op_proto_test SRCS op_proto_test.cc DEPS op_proto protobuf)
cc_test(op_desc_test SRCS op_desc_test.cc DEPS op_desc protobuf)

cc_library(operator SRCS operator.cc DEPS op_desc device_context tensor scope)
cc_test(operator_test SRCS operator_test.cc DEPS operator op_registry)

cc_library(grad_op_builder SRCS grad_op_builder.cc DEPS op_proto operator)
cc_library(op_registry SRCS op_registry.cc DEPS op_desc grad_op_builder)
cc_test(op_registry_test SRCS op_registry_test.cc DEPS op_registry)
cc_test(grad_op_builder_test SRCS grad_op_builder_test.cc DEPS grad_op_builder op_registry add_op)

py_proto_compile(framework_py_proto SRCS attr_type.proto op_proto.proto op_desc.proto)
# Generate an empty __init__.py to make framework_py_proto as a valid python module.
add_custom_target(framework_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)
add_dependencies(framework_py_proto framework_py_proto_init)

cc_library(net SRCS net.cc DEPS op_registry)
cc_test(net_op_test SRCS net_op_test.cc DEPS net)

cc_library(backward SRCS backward.cc DEPS net)
cc_test(backward_test SRCS backward_test.cc DEPS backward)
