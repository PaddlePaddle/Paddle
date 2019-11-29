
set(C_API_SRCS pd_config.cc pd_predictor.cc pd_tensor.cc c_api.cc)

cc_library(paddle_fluid_c SRCS ${C_API_SRCS} DEPS 
            ${fluid_modules} paddle_fluid_api paddle_inference_api analysis_predictor)
cc_library(paddle_fluid_c_shared SHARED SRCS ${C_API_SRCS} DEPS 
            ${fluid_modules} paddle_fluid_api paddle_inference_api analysis_predictor)
set_target_properties(paddle_fluid_c_shared PROPERTIES OUTPUT_NAME paddle_fluid_c)
if(WIN32)
    target_link_libraries(paddle_fluid_c_shared shlwapi.lib)
endif()
