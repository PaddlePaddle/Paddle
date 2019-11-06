
set(C_API_SRCS pd_config.cc pd_predictor.cc pd_tensor.cc c_api.cc)

cc_library(paddle_fluid_c SRCS ${C_API_SRCS} DEPS paddle_fluid)
cc_library(paddle_fluid_c_shared SHARED SRCS ${C_API_SRCS} DEPS paddle_fluid)
set_target_properties(paddle_fluid_c_shared PROPERTIES OUTPUT_NAME paddle_fluid_c)
if(WIN32)
    target_link_libraries(paddle_fluid_c_shared shlwapi.lib)
endif()
