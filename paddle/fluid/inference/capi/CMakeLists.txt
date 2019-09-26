
cc_library(pd_config SRCS pd_config.cc)
cc_library(pd_predictor SRCS pd_predictor.cc)
cc_library(pd_tensor SRCS pd_tensor.cc)

cc_library(paddle_fluid_c SRCS c_api.cc DEPS paddle_fluid pd_config pd_predictor pd_tensor)
#cc_library(paddle_fluid_c_shared SHARED SRCS c_api.cc DEPS paddle_fluid pd_config pd_predictor pd_tensor)
#set_target_properties(paddle_fluid_c_shared PROPERTIES OUTPUT_NAME paddle_fluid_c)
