cc_library(memory_lite SRCS memory.cc)
cc_library(target_wrapper_lite SRCS target_wrapper.cc)
cc_library(tensor_lite SRCS tensor.cc DEPS memory_lite target_wrapper_lite)
cc_library(kernel_lite SRCS kernel.cc DEPS type_system target_wrapper_lite)
cc_library(variable_lite SRCS variable.cc)
cc_library(op_registry_lite SRCS op_registry.cc)
cc_library(scope_lite SRCS scope.cc)
cc_library(op_lite SRCS op_lite.cc DEPS scope_lite op_registry_lite)
cc_library(op_executor_lite SRCS op_executor.cc DEPS scope_lite tensor_lite op_lite op_registry_lite
  #TODO(Superjomn) remove these dependencies from original framework
  proto_desc)
cc_library(kernel_executor_lite SRCS kernel_executor.cc DEPS mir_ssa_graph kernel_lite)
cc_library(types_lite SRCS types.cc)
cc_library(type_system SRCS type_system.cc DEPS tensor_lite)
cc_library(optimizer_lite SRCS optimizer.cc DEPS mir_pass_manager)
cc_library(program_fake_utils SRCS program_fake_utils.cc DEPS mir_ssa_graph
  scope_lite op_registry_lite proto_desc op_lite
  ops_lite
  host_kernels
  )
cc_library(program_lite SRCS program.cc DEPS op_lite kernel_lite)

cc_test(test_scope_lite SRCS scope_test.cc DEPS scope_lite)
cc_test(test_kernel_lite SRCS kernel_test.cc DEPS target_wrapper_x86)
cc_test(test_op_lite SRCS op_lite_test.cc DEPS op_lite)
cc_test(test_tensor_lite SRCS tensor_test.cc)
cc_test(test_op_executor_lite SRCS op_executor_test.cc DEPS op_executor_lite ops_lite host_kernels)
cc_test(test_type_system SRCS type_system_test.cc DEPS type_system)
cc_test(test_optimizer_lite SRCS optimizer_test.cc DEPS mir_pass_manager program_fake_utils mir_passes)
cc_test(test_types_lite SRCS types_test.cc DEPS types_lite)

add_subdirectory(mir)
