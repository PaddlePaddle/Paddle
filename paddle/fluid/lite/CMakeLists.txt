if (NOT WITH_LITE)
    return()
endif()

message(WARNING "Lite enabled!")
message(STATUS "LIGHT_FRAMEWORK:\t${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK}")
message(STATUS "LITE_WITH_CUDA:\t${LITE_WITH_CUDA}")
message(STATUS "LITE_WITH_X86:\t${LITE_WITH_X86}")
message(STATUS "LITE_WITH_ARM:\t${LITE_WITH_ARM}")
message(STATUS "LITE_WITH_PROFILE:\t${LITE_WITH_PROFILE}")

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

function (lite_deps TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS LIGHT_DEPS HVY_DEPS ARGS)
  cmake_parse_arguments(lite_deps "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(deps ${lite_deps_DEPS})

  if(LITE_WITH_X86)
    foreach(var ${lite_deps_X86_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_CUDA)
    foreach(var ${lite_deps_CUDA_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_ARM)
    foreach(var ${lite_deps_ARM_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_PROFILE)
    foreach(var ${lite_deps_PROFILE_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    foreach(var ${lite_deps_LIGHT_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    foreach(var ${lite_deps_HVY_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  set(${TARGET} ${deps} PARENT_SCOPE)

endfunction()

# Add names for lite libraries for latter compile. We use this name list to avoid compiling
# the whole fluid project to accelerate the compile speed.
set(offline_lib_registry_file "${CMAKE_BINARY_DIR}/lite_libs.txt")
file(WRITE ${offline_lib_registry_file} "") # clean
# cc_library with branch support.
# The branches:
#  X86_DEPS: works only when LITE_WITH_X86 is ON.
#  CUDA_DEPS:     LITE_WITH_CUDA
#  ARM_DEPS:      LITE_WITH_ARM
#  PROFILE_DEPS:  LITE_WITH_PROFILE
#  LIGHT_DEPS:    LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#  HVY_DEPS:      NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
function(lite_cc_library TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS LIGHT_DEPS
      HVY_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )

    cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS})

    # register a library name.
    file(APPEND ${offline_lib_registry_file} "${TARGET}\n")
endfunction()

function(lite_cc_binary TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS
      LIGHT_DEPS HVY_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS})
endfunction()

# Add a unit-test name to file for latter offline manual test.
set(offline_test_registry_file "${CMAKE_BINARY_DIR}/lite_tests.txt")
file(WRITE ${offline_test_registry_file} "") # clean
# Test lite modules.
function(lite_cc_test TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS
      LIGHT_DEPS HVY_DEPS
      ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )
    _lite_cc_test(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ARGS ${args_ARGS})
    file(APPEND ${offline_test_registry_file} "${TARGET}\n")
endfunction()

add_subdirectory(core)
add_subdirectory(x86)
add_subdirectory(arm)
add_subdirectory(host)
add_subdirectory(cuda)
add_subdirectory(operators)
add_subdirectory(kernels)
add_subdirectory(model_parser)
add_subdirectory(utils)
add_subdirectory(api)
add_subdirectory(gen_code)
