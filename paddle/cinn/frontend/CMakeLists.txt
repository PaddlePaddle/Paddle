core_gather_headers()
gather_srcs(
  cinnapi_src
  SRCS
  computation.cc
  interpreter.cc
  net_builder.cc
  op_mapper_registry.cc
  program_pass.cc
  optimize.cc)

#cinn_cc_test(test_paddle_model_convertor
#        ARGS --model_dir=${THIRD_PARTY_PATH}/resnet_model
#        SRCS paddle_model_convertor_test.cc DEPS cinncore decomposer_test_helper)

#cinn_cc_test(test_computation
#  ARGS "--model_dir=${THIRD_PARTY_PATH}/naive_mul_model"
#  SRCS computation_test.cc DEPS cinncore)
