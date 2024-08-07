if(WIN32)
  cc_test(
    nccl_context_test
    SRCS nccl_context_test.cc
    DEPS device_context)
else()
  if(WITH_GLOO AND (WITH_NCCL OR WITH_RCCL))
    cc_test(
      nccl_context_test
      SRCS nccl_context_test.cc
      DEPS nccl_context)
    cc_test(
      heter_ccl_context_test
      SRCS heter_ccl_context_test.cc
      DEPS heter_ccl_context
           nccl_context
           imperative_gloo_context
           gloo_context
           gloo_wrapper
           gloo
           framework_io)
    #set_tests_properties(heter_ccl_context_test PROPERTIES LABELS "RUN_TYPE=DIST")
  endif()
  if(WITH_XPU_BKCL)
    cc_test(
      bkcl_context_test
      SRCS bkcl_context_test.cc
      DEPS bkcl_context)
  endif()
endif()

cc_test(
  test_gradient_accmulator
  SRCS test_gradient_accmulator.cc
  DEPS selected_rows_utils gradient_accumulator phi common phi_utils)
cc_test(
  test_layer
  SRCS test_layer.cc
  DEPS layer proto_desc operator op_registry variable_helper generated_op)
cc_test(
  test_prepare_op
  SRCS test_prepare_op.cc
  DEPS prepared_operator
       op_info
       split_op
       layer
       activation_op
       phi
       common)
cc_test(
  test_tracer
  SRCS test_tracer.cc
  DEPS tracer
       layer
       proto_desc
       operator
       op_registry
       variable_helper
       generated_op
       generated_static_op
       elementwise_add_op)
cc_test(
  test_hooks
  SRCS test_hooks.cc
  DEPS tracer
       basic_engine
       layer
       proto_desc
       operator
       op_registry
       variable_helper
       generated_op
       elementwise_add_op)
cc_test(
  test_eager
  SRCS test_eager.cc
  DEPS tracer layer prepared_operator generated_op)
if(WITH_NCCL
   OR WITH_RCCL
   OR WITH_XPU_BKCL)
  cc_test(
    test_group
    SRCS test_group.cc
    DEPS reducer phi common)
endif()
