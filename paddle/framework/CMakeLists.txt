# ddim lib
cc_library(ddim SRCS ddim.cc DEPS eigen3)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

cc_library(tensor SRCS tensor.cc DEPS ddim place paddle_memory device_context)
cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)
cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

cc_library(lod_tensor SRCS lod_tensor.cc details/lod_tensor.cc DEPS ddim place tensor)
cc_test(lod_tensor_test SRCS lod_tensor_test.cc DEPS lod_tensor)

cc_test(variable_test SRCS variable_test.cc)

cc_library(scope SRCS scope.cc)
cc_test(scope_test SRCS scope_test.cc DEPS scope)

proto_library(framework_proto SRCS framework.proto)

cc_library(attribute SRCS attribute.cc DEPS framework_proto)

cc_library(operator SRCS operator.cc DEPS framework_proto device_context tensor scope attribute)
cc_test(operator_test SRCS operator_test.cc DEPS operator op_registry)

cc_library(grad_op_builder SRCS grad_op_builder.cc DEPS operator)
cc_library(op_registry SRCS op_registry.cc DEPS grad_op_builder)
cc_test(op_registry_test SRCS op_registry_test.cc DEPS op_registry)
cc_test(grad_op_builder_test SRCS grad_op_builder_test.cc DEPS grad_op_builder op_registry add_op)

py_proto_compile(framework_py_proto SRCS framework.proto)
# Generate an empty __init__.py to make framework_py_proto as a valid python module.
add_custom_target(framework_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)
add_dependencies(framework_py_proto framework_py_proto_init)

cc_library(backward SRCS backward.cc DEPS net_op)
cc_test(backward_test SRCS backward_test.cc DEPS backward)

if(WITH_PYTHON)
cc_library(paddle_pybind SHARED
    SRCS pybind.cc
    DEPS pybind python backward
    fc_op
    sgd_op
    add_op
    mean_op
    cross_entropy_op
    recurrent_op
    uniform_random_op
    fill_zeros_like_op)
endif(WITH_PYTHON)
