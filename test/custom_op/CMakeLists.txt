if(WITH_TESTING)
  # New custom OP can support Windows/Linux/Mac now
  if(WITH_GPU OR APPLE)
    py_test(test_custom_relu_op_setup SRCS test_custom_relu_op_setup.py)
    py_test(test_custom_relu_op_jit SRCS test_custom_relu_op_jit.py)
    py_test(test_custom_relu_model SRCS test_custom_relu_model.py)
    py_test(test_context_pool SRCS test_context_pool.py)

    # Compiling shared library will cost some time, but running process is very fast.
    set_tests_properties(test_custom_relu_op_setup PROPERTIES TIMEOUT 250)
    set_tests_properties(test_custom_relu_op_jit PROPERTIES TIMEOUT 180)
    set_tests_properties(test_custom_relu_model PROPERTIES TIMEOUT 180)
    set_tests_properties(test_context_pool PROPERTIES TIMEOUT 180)
    py_test(test_custom_cast_op_jit SRCS test_custom_cast_op_jit.py)
    set_tests_properties(test_custom_cast_op_jit PROPERTIES TIMEOUT 180)
  endif()

  if(NOT WIN32)
    # TODO(YuanRisheng) : Currently, we run this unittest by translating old ir to new ir, and it has bug that can't judge whether op_desc is a inplace op in windows.
    # We will fix it when abandoning translation in final state.
    if(WITH_GPU)
      py_test(test_inference_inplace_pir SRCS test_inference_inplace_pir.py
                                              ENVS FLAGS_enable_pir_api=1)
      set_tests_properties(test_inference_inplace_pir PROPERTIES TIMEOUT 180)
    endif()
  endif()

  # custom OP support TensorRT inference
  if(WITH_GPU
     AND WITH_TENSORRT
     AND (NOT WITH_COVERAGE))
    py_test(test_inference_gap_setup SRCS test_inference_gap_setup.py)

    # Compiling shared library will cost some time, but running process is very fast.
    set_tests_properties(test_inference_gap_setup PROPERTIES TIMEOUT 180)
  endif()

  if(WITH_GPU AND WITH_DISTRIBUTE)
    py_test(test_custom_op_relu_model_static_multidevice
            SRCS test_custom_op_relu_model_static_multidevice.py)
    set_tests_properties(test_custom_op_relu_model_static_multidevice
                         PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 180)
  endif()
  if(WITH_XPU AND WITH_DISTRIBUTE)
    set(CUSTOM_XPU_ENVS FLAGS_init_allocated_mem=0)
    py_test(test_custom_relu_op_xpu_setup SRCS test_custom_relu_op_xpu_setup.py
                                               ENVS ${CUSTOM_XPU_ENVS})
    py_test(test_custom_op_relu_model_static_multidevice
            SRCS test_custom_op_relu_model_static_multidevice.py ENVS
                 ${CUSTOM_XPU_ENVS})
    set_tests_properties(test_custom_op_relu_model_static_multidevice
                         PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 180)
  endif()

  py_test(test_custom_tensor_operator SRCS test_custom_tensor_operator.py)
  set_tests_properties(test_custom_tensor_operator PROPERTIES TIMEOUT 180)

  # CPU custom op tests: only compile .cc file
  py_test(test_dispatch_jit SRCS test_dispatch_jit.py)
  py_test(test_multi_out_jit SRCS test_multi_out_jit.py)
  py_test(test_custom_attrs_jit SRCS test_custom_attrs_jit.py)
  py_test(test_custom_extend_attrs_jit SRCS test_custom_extend_attrs_jit.py)
  set_tests_properties(test_custom_extend_attrs_jit PROPERTIES TIMEOUT 180)
  py_test(test_custom_concat SRCS test_custom_concat.py)
  set_tests_properties(
    test_custom_concat PROPERTIES ENVIRONMENT
                                  FLAGS_new_executor_static_build=true)
  py_test(test_custom_conj SRCS test_custom_conj.py)
  py_test(test_custom_linear SRCS test_custom_linear.py)
  py_test(test_custom_simple_slice SRCS test_custom_simple_slice.py)
  py_test(test_custom_tanh_double_grad SRCS test_custom_tanh_double_grad.py)
  py_test(test_custom_inplace SRCS test_custom_inplace.py)
  py_test(test_custom_optional SRCS test_custom_optional.py)
  py_test(test_custom_contiguous SRCS test_custom_contiguous.py)
  set_tests_properties(test_custom_contiguous PROPERTIES TIMEOUT 180)

  # other tests
  py_test(test_sysconfig SRCS test_sysconfig.py)
  py_test(test_check_abi SRCS test_check_abi.py)

  if(APPLE)
    set_tests_properties(test_custom_simple_slice PROPERTIES TIMEOUT 300)
  endif()
endif()
