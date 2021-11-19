cc_test(test_allocator SRCS test_allocator.cc DEPS tensor_base)
cc_test(test_storage SRCS test_storage.cc DEPS tensor_base)
cc_test(test_dense_tensor SRCS test_dense_tensor.cc DEPS dense_tensor)
cc_test(test_intrusive_ptr SRCS test_intrusive_ptr.cc)
cc_test(test_type_info SRCS test_type_info.cc)
cc_test(test_convert_utils SRCS test_convert_utils.cc DEPS convert_utils)
cc_test(test_kernel_factory SRCS test_kernel_factory.cc DEPS kernel_factory)

if(WITH_ARM)
  # TODO(wilber): support arm_context
  # set(pten_ctx arm_context)
else()
  set(pten_ctx x86_context)
endif()

if(WITH_GPU)
  set(pten_ctx ${pten_ctx} cuda_context)
elseif(WITH_ROCM)
  set(pten_ctx ${pten_ctx} rocm_context)
endif()

if (WITH_XPU)
  set(pten_ctx ${pten_ctx} xpu_context)
endif()

cc_test(test_context SRCS test_context.cc DEPS pten_device_context ${pten_ctx})
