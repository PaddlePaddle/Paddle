
set(C_API_SRCS pd_config.cc pd_predictor.cc pd_tensor.cc c_api.cc)
if(WIN32)
    cc_library(paddle_fluid_c_shared SHARED SRCS ${C_API_SRCS} DEPS ${paddle_fluid_list})
    cc_library(paddle_fluid_c SRCS ${C_API_SRCS} DEPS ${paddle_fluid_list})
else()
    cc_library(paddle_fluid_c SRCS ${C_API_SRCS} DEPS paddle_fluid)
    cc_library(paddle_fluid_c_shared SHARED SRCS ${C_API_SRCS} DEPS paddle_fluid)
endif()
set_target_properties(paddle_fluid_c_shared PROPERTIES OUTPUT_NAME paddle_fluid_c)
if(WIN32)
    target_link_libraries(paddle_fluid_c_shared shlwapi.lib)
endif()

# capi-inference-only library

set(FLUID_INFERENCE_C_INSTALL_DIR "${CMAKE_BINARY_DIR}/fluid_inference_c_install_dir" CACHE STRING
"A path setting CAPI fluid inference shared")
copy_part_of_thrid_party(paddle_fluid_c_shared ${FLUID_INFERENCE_C_INSTALL_DIR})

set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
if(WIN32)
    set(paddle_fluid_c_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/capi/${CMAKE_BUILD_TYPE}/paddle_fluid_c.dll
        ${PADDLE_BINARY_DIR}/paddle/fluid/inference/capi/${CMAKE_BUILD_TYPE}/paddle_fluid_c.lib)
else(WIN32)
    set(paddle_fluid_c_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/libpaddle_fluid.*)
endif(WIN32)

if(WIN32)
    copy(paddle_fluid_c_shared
            SRCS  ${src_dir}/inference/capi/c_api.h ${paddle_fluid_c_lib}
            DSTS  ${FLUID_INFERENCE_C_INSTALL_DIR}/paddle/include ${FLUID_INFERENCE_C_INSTALL_DIR}/paddle/lib
                  ${FLUID_INFERENCE_C_INSTALL_DIR}/paddle/lib)
else()
    copy(paddle_fluid_c_shared
        SRCS  ${src_dir}/inference/capi/c_api.h ${paddle_fluid_c_lib}
        DSTS  ${FLUID_INFERENCE_C_INSTALL_DIR}/paddle/include ${FLUID_INFERENCE_C_INSTALL_DIR}/paddle/lib)
endif()

version(${FLUID_INFERENCE_C_INSTALL_DIR}/version.txt)
