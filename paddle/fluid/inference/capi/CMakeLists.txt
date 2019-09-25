
cc_library(pd_config SRCS pd_config.cc)
cc_library(pd_predictor SRCS pd_predictor.cc)
cc_library(pd_tensor SRCS pd_tensor.cc)

cc_library(c_api SRCS c_api.cc DEPS paddle_fluid pd_config pd_predictor pd_tensor)
