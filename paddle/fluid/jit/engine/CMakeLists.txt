cc_library(
  jit_interpreter_engine
  SRCS interpreter_engine.cc
  DEPS)

cc_library(
  jit_predictor_engine
  SRCS predictor_engine.cc
  DEPS paddle_inference_api analysis_predictor)
