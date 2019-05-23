if (NOT WITH_LITE)
    return()
endif()

message(WARNING "Lite enabled!")
message(STATUS "LIGHT_FRAMEWORK:\t${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK}")
message(STATUS "LITE_WITH_CUDA:\t${LITE_WITH_CUDA}")
message(STATUS "LITE_WITH_X86:\t${LITE_WITH_X86}")
message(STATUS "LITE_WITH_ARM:\t${LITE_WITH_ARM}")

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

function (lite_deps DEPS)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEPS X86_DEPS CUDA_DEPS ARM_DEPS)
  cmake_parse_arguments(lite_deps "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(${DEPS} ${lite_deps_DEPS} PARENT_SCOPE)
  foreach(var ${lite_deps_X86_DEPS})
    set(${DEPS} ${${DEPS}} ${var} PARENT_SCOPE)
  endforeach()

  foreach(var ${lite_deps_CUDA_DEPS})
    set(${DEPS} ${${DEPS}} ${var} PARENT_SCOPE)
  endforeach()

  foreach(var ${lite_deps_ARM_DEPS})
    set(${DEPS} ${${DEPS}} ${var} PARENT_SCOPE)
  endforeach()
endfunction()

function(lite_cc_library TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS ARM_DEPS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            )

    cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps})
endfunction()

function(lite_cc_test TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS ARM_DEPS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            )
    _lite_cc_test(${TARGET} SRCS ${args_SRCS} DEPS ${deps})
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
