if (WITH_TESTING)
    lite_cc_library(lite_gtest_main SRCS lite_gtest_main.cc DEPS gtest gflags)
endif()
lite_cc_library(target_wrapper_lite SRCS target_wrapper.cc
  DEPS target_wrapper_host place_lite
  X86_DEPS target_wrapper_x86
  CUDA_DEPS target_wrapper_cuda)
lite_cc_library(memory_lite SRCS memory.cc DEPS target_wrapper_lite)
lite_cc_library(lite_tensor SRCS lite_tensor.cc DEPS memory_lite target_wrapper_lite)
if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    lite_cc_library(hvy_tensor SRCS hvy_tensor.cc DEPS lod_tensor HVY_DEPS framework_proto)
endif()

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  set(tensor_lite lite_tensor)
else()
  set(tensor_lite hvy_tensor)
endif()

proto_library(framework_proto_lite SRCS framework.proto)

if (LITE_WITH_X86)
lite_cc_library(variable_lite SRCS variable.cc DEPS framework_proto)
lite_cc_library(types_lite SRCS types.cc DEPS framework_proto)
else()
lite_cc_library(variable_lite SRCS variable.cc)
lite_cc_library(types_lite SRCS types.cc)
endif()
lite_cc_library(op_registry_lite SRCS op_registry.cc DEPS framework_proto_lite)
lite_cc_library(scope_lite SRCS scope.cc DEPS ${tensor_lite})
lite_cc_library(cpu_info_lite SRCS cpu_info.cc)

if (LITE_WITH_ARM)
lite_cc_library(context_lite SRCS context.cc DEPS ${tensor_lite} any_lite cpu_info_lite CL_DEPS cl_helper)
else()
lite_cc_library(context_lite SRCS context.cc DEPS ${tensor_lite} any_lite cpu_info_lite eigen3 CL_DEPS cl_helper)
endif()
lite_cc_library(kernel_lite SRCS kernel.cc DEPS context_lite type_system target_wrapper_lite any_lite op_params_lite framework_proto_lite ${tensor_lite})
lite_cc_library(op_lite SRCS op_lite.cc DEPS scope_lite op_registry_lite target_wrapper_lite kernel_lite
  cpp_op_desc_lite ${tensor_lite})
lite_cc_library(type_system SRCS type_system.cc DEPS ${tensor_lite} target_wrapper_lite)

lite_cc_library(program_lite SRCS program.cc
    DEPS op_lite kernel_lite compatible_pb_lite model_parser_lite ${ops_lite}
    HVY_DEPS framework_proto
    PROFILE_DEPS basic_profiler_lite)
lite_cc_library(optimizer_lite SRCS optimizer.cc DEPS mir_pass_manager model_parser_lite program_lite)

add_subdirectory(mir)
add_subdirectory(profile)

# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

lite_cc_library(program_fake_utils SRCS program_fake_utils.cc DEPS mir_ssa_graph
        scope_lite op_registry_lite proto_desc op_lite
        ${ops_lite}
        ${host_kernels}
        )

lite_cc_test(test_scope_lite SRCS scope_test.cc DEPS scope_lite)
lite_cc_test(test_kernel_lite SRCS kernel_test.cc DEPS kernel_lite target_wrapper_lite any_lite)
lite_cc_test(test_op_lite SRCS op_lite_test.cc DEPS op_lite)
lite_cc_test(test_tensor_lite SRCS lite_tensor_test.cc DEPS lite_tensor)
lite_cc_test(test_type_system SRCS type_system_test.cc DEPS type_system utils_lite)
#lite_cc_test(test_optimizer_lite SRCS optimizer_test.cc DEPS mir_pass_manager program_fake_utils mir_passes optimizer_lite fc_op_lite)
lite_cc_test(test_types_lite SRCS types_test.cc DEPS types_lite)
lite_cc_test(test_memory_lite SRCS memory_test.cc DEPS memory_lite)
lite_cc_test(test_context_lite SRCS context_test.cc DEPS context_lite X86_DEPS operator)
