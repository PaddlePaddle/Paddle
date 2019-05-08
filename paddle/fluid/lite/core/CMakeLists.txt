cc_library(lite_gtest_main SRCS lite_gtest_main.cc)
cc_library(memory_lite SRCS memory.cc)
cc_library(target_wrapper_lite SRCS target_wrapper.cc)

cc_library(lite_tensor SRCS lite_tensor.cc DEPS memory_lite target_wrapper_lite)
if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    cc_library(hvy_tensor SRCS hvy_tensor.cc DEPS lod_tensor)
endif()

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  set(tensor_lite lite_tensor)
else()
  set(tensor_lite hvy_tensor)
endif()

proto_library(framework_proto SRCS framework.proto)

cc_library(kernel_lite SRCS kernel.cc DEPS type_system target_wrapper_lite)
cc_library(variable_lite SRCS variable.cc)
cc_library(op_registry_lite SRCS op_registry.cc)
cc_library(scope_lite SRCS scope.cc)
cc_library(op_lite SRCS op_lite.cc DEPS scope_lite op_registry_lite compatible_pb_lite)
cc_library(types_lite SRCS types.cc)
cc_library(type_system SRCS type_system.cc DEPS ${tensor_lite})

cc_library(kernel_executor_lite SRCS kernel_executor.cc DEPS mir_ssa_graph kernel_lite)
cc_library(program_lite SRCS program.cc DEPS op_lite kernel_lite)
cc_library(optimizer_lite SRCS optimizer.cc DEPS mir_pass_manager model_parser_lite program_lite)

add_subdirectory(mir)

# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

cc_library(program_fake_utils SRCS program_fake_utils.cc DEPS mir_ssa_graph
        scope_lite op_registry_lite proto_desc op_lite
        ops_lite
        host_kernels
        )

lite_cc_test(test_scope_lite SRCS scope_test.cc DEPS scope_lite)
lite_cc_test(test_kernel_lite SRCS kernel_test.cc DEPS kernel_lite target_wrapper_x86)
lite_cc_test(test_op_lite SRCS op_lite_test.cc DEPS op_lite)
lite_cc_test(test_tensor_lite SRCS lite_tensor_test.cc DEPS lite_tensor)
lite_cc_test(test_type_system SRCS type_system_test.cc DEPS type_system)
lite_cc_test(test_optimizer_lite SRCS optimizer_test.cc DEPS mir_pass_manager program_fake_utils mir_passes)
lite_cc_test(test_types_lite SRCS types_test.cc DEPS types_lite)
