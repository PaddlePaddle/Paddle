if (NOT WITH_LITE)
    return()
endif()

message(WARNING "Lite enabled!")
message(STATUS "LIGHT_FRAMEWORK: ${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK}")
message(STATUS "LITE_WITH_CUDA: ${LITE_WITH_CUDA}")
message(STATUS "LITE_WITH_X86: ${LITE_WITH_X86}")

set(LITE_MODEL_DIR "${THIRD_PARTY_PATH}/install")

function(lite_download_and_uncompress INSTALL_DIR URL FILENAME)
    message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
    string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
    set(EXTERNAL_PROJECT_NAME "extern_lite_download_${FILENAME_EX}")
    set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
    ExternalProject_Add(
            ${EXTERNAL_PROJECT_NAME}
            ${EXTERNAL_PROJECT_LOG_ARGS}
            PREFIX                ${INSTALL_DIR}
            DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME} &&
            ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILENAME}
            DOWNLOAD_DIR          ${INSTALL_DIR}
            DOWNLOAD_NO_PROGRESS  1
            CONFIGURE_COMMAND     ""
            BUILD_COMMAND         ""
            UPDATE_COMMAND        ""
            INSTALL_COMMAND       ""
    )
endfunction()


add_subdirectory(core)
add_subdirectory(x86)
add_subdirectory(host)
add_subdirectory(cuda)
add_subdirectory(operators)
add_subdirectory(kernels)
add_subdirectory(model_parser)
add_subdirectory(utils)
add_subdirectory(api)
