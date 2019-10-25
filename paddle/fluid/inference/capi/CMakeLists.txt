
set(C_API_SRCS pd_config.cc pd_predictor.cc pd_tensor.cc c_api.cc)
if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
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

if(WIN32)
    if(NOT PYTHON_EXECUTABLE)
        FIND_PACKAGE(PythonInterp REQUIRED)
    endif()
endif()

set(COPY_SCRIPT_DIR ${PADDLE_SOURCE_DIR}/cmake)
function(copy TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DSTS)
    cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
    list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
    if (NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
        message("${copy_lib_SRCS_len}============${copy_lib_DSTS_len}")
        message(FATAL_ERROR "${TARGET} source numbers are not equal to destination numbers")
    endif ()
    math(EXPR len "${copy_lib_SRCS_len} - 1")
    foreach (index RANGE ${len})
        list(GET copy_lib_SRCS ${index} src)
        list(GET copy_lib_DSTS ${index} dst)
        if (WIN32)   #windows
            file(TO_NATIVE_PATH ${src} native_src)
            file(TO_NATIVE_PATH ${dst} native_dst)
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${PYTHON_EXECUTABLE} ${COPY_SCRIPT_DIR}/copyfile.py ${native_src} ${native_dst})
        else (WIN32) #not windows
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND mkdir -p "${dst}"
                    COMMAND cp -r "${src}" "${dst}"
                    COMMENT "copying ${src} -> ${dst}")
        endif (WIN32) # not windows
    endforeach ()
endfunction()

# capi-inference-only library
set(FLUID_INFERENCE_C_INSTALL_DIR "${CMAKE_BINARY_DIR}/fluid_inference_c_install_dir" CACHE STRING
"A path setting CAPI fluid inference shared")

if(WITH_MKLML)
    set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/mklml")
    if(WIN32)
        copy(paddle_fluid_c_shared
                SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_SHARED_LIB}
                ${MKLML_SHARED_LIB_DEPS} ${MKLML_SHARED_IOMP_LIB} ${MKLML_INC_DIR}
                DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}/lib
                ${dst_dir}/lib ${dst_dir}/lib ${dst_dir})
    else()
        copy(paddle_fluid_c_shared
                SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_INC_DIR}
                DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir})
    endif()
elseif (NOT CBLAS_FOUND OR WIN32)
    set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/openblas")
    copy(paddle_fluid_c_shared
            SRCS ${CBLAS_INSTALL_DIR}/lib ${CBLAS_INSTALL_DIR}/include
            DSTS ${dst_dir} ${dst_dir})
endif ()

if(WITH_MKLDNN)
    set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/mkldnn")
    if(WIN32)
        copy(paddle_fluid_c_shared
                SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB} ${MKLDNN_LIB}
                DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir}/lib)
    else()
        copy(paddle_fluid_c_shared
                SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB}
                DSTS ${dst_dir} ${dst_dir}/lib)
    endif()
endif()

set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/gflags")
copy(paddle_fluid_c_shared
        SRCS ${GFLAGS_INCLUDE_DIR} ${GFLAGS_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib)

set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/glog")
copy(paddle_fluid_c_shared
        SRCS ${GLOG_INCLUDE_DIR} ${GLOG_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib)

if (NOT PROTOBUF_FOUND OR WIN32)
    set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/protobuf")
    copy(paddle_fluid_c_shared
            SRCS ${PROTOBUF_INCLUDE_DIR} ${PROTOBUF_LIBRARY}
            DSTS ${dst_dir} ${dst_dir}/lib)
endif ()

if (WITH_NGRAPH)
    set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/ngraph")
    copy(paddle_fluid_c_shared
            SRCS ${NGRAPH_INC_DIR} ${NGRAPH_LIB_DIR}
            DSTS ${dst_dir} ${dst_dir})
endif ()

if (TENSORRT_FOUND)
    set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/tensorrt")
    copy(paddle_fluid_c_shared
            SRCS ${TENSORRT_ROOT}/include/Nv*.h ${TENSORRT_ROOT}/lib/*nvinfer*
            DSTS ${dst_dir}/include ${dst_dir}/lib)
endif ()

if (ANAKIN_FOUND)
    set(dst_dir "${FLUID_INFERENCE_C_INSTALL_DIR}/third_party/install/anakin")
    copy(paddle_fluid_c_shared
            SRCS ${ANAKIN_ROOT}/*
            DSTS ${dst_dir})
endif ()

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
