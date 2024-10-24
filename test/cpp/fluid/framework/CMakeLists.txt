add_subdirectory(details)

paddle_test(data_type_test SRCS data_type_test.cc)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(data_type_test)
endif()

nv_test(
  tensor_test
  SRCS tensor_test.cc
  DEPS tensor)
if(WITH_GPU)
  nv_test(
    tensor_util_test
    SRCS tensor_util_test.cc tensor_util_test.cu
    DEPS tensor dlpack_tensor)
elseif(WITH_ROCM)
  hip_test(
    tensor_util_test
    SRCS tensor_util_test.cc tensor_util_test.cu
    DEPS tensor dlpack_tensor)
else()
  nv_test(
    tensor_util_test
    SRCS tensor_util_test.cc
    DEPS tensor dlpack_tensor)
endif()

nv_test(
  copy_same_tensor_test
  SRCS copy_same_tensor_test.cc
  DEPS tensor)

paddle_test(eigen_test SRCS eigen_test.cc)

paddle_test(lod_tensor_test SRCS lod_tensor_test.cc DEPS common)

if(WITH_GPU)
  nv_test(
    lod_tensor_gpu_test
    SRCS lod_tensor_test.cu
    DEPS lod_tensor)
elseif(WITH_ROCM)
  hip_test(
    lod_tensor_gpu_test
    SRCS lod_tensor_test.cu
    DEPS lod_tensor)
endif()

paddle_test(reader_test SRCS reader_test.cc)

paddle_test(threadpool_test SRCS threadpool_test.cc DEPS common)

paddle_test(var_type_traits_test SRCS var_type_traits_test.cc)

paddle_test(device_worker_test SRCS device_worker_test.cc)

paddle_test(scope_test SRCS scope_test.cc)

paddle_test(variable_test SRCS variable_test.cc)

if(WITH_GPU)
  nv_test(
    data_device_transform_test
    SRCS data_device_transform_test.cu
    DEPS operator op_registry device_context phi common scope)
elseif(WITH_ROCM)
  hip_test(
    data_device_transform_test
    SRCS data_device_transform_test.cu
    DEPS operator op_registry device_context phi common scope)
endif()

if(WITH_GPU)
  nv_test(
    data_type_transform_test
    SRCS data_type_transform_test.cc data_type_transform_test.cu
    DEPS data_type_transform)
elseif(WITH_ROCM)
  hip_test(
    data_type_transform_test
    SRCS data_type_transform_test.cc data_type_transform_test.cu
    DEPS data_type_transform)
elseif(WITH_XPU)
  paddle_test(data_type_transform_test SRCS data_type_transform_test.cc)
else()
  paddle_test(data_type_transform_test SRCS data_type_transform_test.cc)
endif()

paddle_test(data_layout_transform_test SRCS data_layout_transform_test.cc)

paddle_test(attribute_test SRCS attribute_test.cc)

paddle_test(program_desc_test SRCS program_desc_test.cc)

paddle_test(op_desc_test SRCS op_desc_test.cc)

cc_test(
  op_version_registry_test
  SRCS op_version_registry_test.cc
  DEPS op_version_registry)

cc_test(
  op_proto_maker_test
  SRCS op_proto_maker_test.cc
  DEPS op_proto_maker)

cc_test(
  no_need_buffer_vars_inference_test
  SRCS no_need_buffer_vars_inference_test.cc
  DEPS no_need_buffer_vars_inference layer)

cc_test(
  operator_test
  SRCS operator_test.cc
  DEPS operator op_registry device_context)
cc_test(
  operator_exception_test
  SRCS operator_exception_test.cc
  DEPS operator op_registry device_context)

cc_test(
  version_test
  SRCS version_test.cc
  DEPS version)

cc_test(
  op_call_stack_test
  SRCS op_call_stack_test.cc
  DEPS op_call_stack)

cc_test(
  program_utils_test
  SRCS program_utils_test.cc
  DEPS proto_desc program_utils)

if(WITH_GPU)
  nv_test(
    op_registry_test
    SRCS op_registry_test.cc
    DEPS op_registry)
elseif(WITH_ROCM)
  hip_test(
    op_registry_test
    SRCS op_registry_test.cc
    DEPS op_registry)
endif()

if(WITH_PSCORE)
  get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)
  if(WITH_HETERPS)
    cc_test(
      dist_multi_trainer_test
      SRCS dist_multi_trainer_test.cc
      DEPS conditional_block_op executor gloo_wrapper ${RPC_DEPS}
           graph_gpu_wrapper)
    cc_test(
      heter_pipeline_trainer_test
      SRCS heter_pipeline_trainer_test.cc
      DEPS conditional_block_op
           generated_op
           heter_listen_and_serv_op
           executor
           heter_server
           gloo_wrapper
           phi
           common
           ${RPC_DEPS}
           graph_gpu_wrapper)
  else()
    cc_test(
      dist_multi_trainer_test
      SRCS dist_multi_trainer_test.cc
      DEPS conditional_block_op executor gloo_wrapper ${RPC_DEPS})
    cc_test(
      heter_pipeline_trainer_test
      SRCS heter_pipeline_trainer_test.cc
      DEPS conditional_block_op
           generated_op
           heter_listen_and_serv_op
           executor
           heter_server
           gloo_wrapper
           phi
           common
           ${RPC_DEPS})
  endif()
else()
  cc_test(
    dist_multi_trainer_test
    SRCS dist_multi_trainer_test.cc
    DEPS conditional_block_op executor gloo_wrapper)
endif()

cc_test(
  prune_test
  SRCS prune_test.cc
  DEPS op_info prune device_context)
cc_test(
  var_type_inference_test
  SRCS var_type_inference_test.cc
  DEPS op_registry proto_desc)

cc_test(
  selected_rows_utils_test
  SRCS selected_rows_utils_test.cc
  DEPS selected_rows_utils)

cc_test(
  op_kernel_type_test
  SRCS op_kernel_type_test.cc
  DEPS phi common device_context framework_proto op_kernel_type)

cc_test(tuple_test SRCS tuple_test.cc)

cc_test(inlined_vector_test SRCS inlined_vector_test.cc)

cc_test(
  dlpack_tensor_test
  SRCS dlpack_tensor_test.cc
  DEPS dlpack_tensor glog)

cc_test(
  op_compatible_info_test
  SRCS op_compatible_info_test.cc
  DEPS op_compatible_info proto_desc string_helper glog)

cc_test(
  infershape_utils_test
  SRCS infershape_utils_test.cc
  DEPS operator phi)

if(WITH_TESTING AND TEST selected_rows_utils_test)
  set_tests_properties(selected_rows_utils_test PROPERTIES TIMEOUT 120)
endif()

cc_test(scope_guard_test SRCS scope_guard_test.cc)
cc_test(
  phi_utils_test
  SRCS phi_utils_test.cc
  DEPS phi_utils)

cc_test(convert_utils_test SRCS convert_utils_test.cc)

cc_test(
  test_fs
  SRCS io/test_fs.cc
  DEPS framework_io string_helper)

if(WITH_CRYPTO)
  cc_test(
    aes_cipher_test
    SRCS io/aes_cipher_test.cc
    DEPS framework_io)
  cc_test(
    cipher_utils_test
    SRCS io/cipher_utils_test.cc
    DEPS framework_io)
endif()

cc_test(
  test_fleet_cc
  SRCS fleet/test_fleet.cc
  DEPS fleet_wrapper gloo_wrapper framework_io string_helper)

cc_test(
  workqueue_test
  SRCS new_executor/workqueue_test.cc
  DEPS standalone_executor)
