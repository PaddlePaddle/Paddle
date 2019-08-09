if (NOT WITH_LITE)
    return()
endif()

include(lite)

message(WARNING "Lite enabled!")
message(STATUS "LIGHT_FRAMEWORK:\t${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK}")
message(STATUS "LITE_WITH_CUDA:\t${LITE_WITH_CUDA}")
message(STATUS "LITE_WITH_X86:\t${LITE_WITH_X86}")
message(STATUS "LITE_WITH_ARM:\t${LITE_WITH_ARM}")
message(STATUS "LITE_WITH_PROFILE:\t${LITE_WITH_PROFILE}")

set(LITE_MODEL_DIR "${THIRD_PARTY_PATH}/install")

set(LITE_ON_MOBILE ${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK})

set(LITE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")

function(lite_download_and_uncompress INSTALL_DIR URL FILENAME)
    message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
    string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
    set(EXTERNAL_PROJECT_NAME "extern_lite_download_${FILENAME_EX}")
    set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
    ExternalProject_Add(
            ${EXTERNAL_PROJECT_NAME}
            ${EXTERNAL_PROJECT_LOG_ARGS}
            PREFIX                ${INSTALL_DIR}
            DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME} && ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILENAME}
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
  set(multiValueArgs DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS LIGHT_DEPS HVY_DEPS CL_DEPS ARGS)
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

  if (LITE_WITH_OPENCL)
    foreach(var ${lite_deps_CL_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  set(${TARGET} ${deps} PARENT_SCOPE)
endfunction()


# A fake target to include all the libraries and tests the lite module depends.
add_custom_target(lite_compile_deps COMMAND echo 1)

# Add names for lite libraries for latter compile. We use this name list to avoid compiling
# the whole fluid project to accelerate the compile speed.
set(offline_lib_registry_file "${CMAKE_BINARY_DIR}/lite_libs.txt")
file(WRITE ${offline_lib_registry_file} "") # clean

set(__lite_cc_files "" ; "")
set(__lite_cc_files "${CMAKE_BINARY_DIR}/lite_cc_files.txt")
file(WRITE ${__lite_cc_files} "") # clean


# cc_library with branch support.
# The branches:
#  X86_DEPS: works only when LITE_WITH_X86 is ON.
#  CUDA_DEPS:     LITE_WITH_CUDA
#  ARM_DEPS:      LITE_WITH_ARM
#  PROFILE_DEPS:  LITE_WITH_PROFILE
#  LIGHT_DEPS:    LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#  HVY_DEPS:      NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#  EXCLUDE_COMPILE_DEPS: TARGET will not be included in lite_compile_deps if this is not None
function(lite_cc_library TARGET)
    set(options SHARED shared STATIC static MODULE module)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS PROFILE_DEPS LIGHT_DEPS
      HVY_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )

    if (args_SHARED OR ARGS_shared)
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS} SHARED)
    elseif (args_MODULE OR ARGS_module)
        add_library(${TARGET} MODULE ${args_SRCS})
        add_dependencies(${TARGET} ${deps} ${args_DEPS})
    else()
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS})
    endif()
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)

    foreach(cc_file ${args_SRCS})
        file(APPEND ${__lite_cc_files} "${cc_file}\n")
    endforeach()

    # collect targets need to compile for lite
    if (args_SRCS AND NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()

    # register a library name.
    file(APPEND ${offline_lib_registry_file} "${TARGET}\n")
endfunction()

function(lite_cc_binary TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS PROFILE_DEPS
      LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS})
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    # collect targets need to compile for lite
    if (NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()
endfunction()

# Add a unit-test name to file for latter offline manual test.
set(offline_test_registry_file "${CMAKE_BINARY_DIR}/lite_tests.txt")
file(WRITE ${offline_test_registry_file} "") # clean
# Test lite modules.
function(lite_cc_test TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS PROFILE_DEPS
      LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS
      ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )
    _lite_cc_test(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ARGS ${args_ARGS})
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    file(APPEND ${offline_test_registry_file} "${TARGET}\n")

    # collect targets need to compile for lite
    if (NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()
endfunction()

add_subdirectory(operators)
add_subdirectory(kernels)
add_subdirectory(core)
add_subdirectory(x86)
add_subdirectory(arm)
add_subdirectory(host)
add_subdirectory(cuda)
add_subdirectory(opencl)
add_subdirectory(model_parser)
add_subdirectory(utils)
add_subdirectory(api)
add_subdirectory(gen_code)
add_subdirectory(tools)

if (WITH_TESTING)
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "lite_naive_model.tar.gz")
    if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
	lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v2_relu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "inception_v4_simple.tar.gz")
    endif()
    if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "GoogleNet_inference.tar.gz")
	    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v2_relu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "inception_v4_simple.tar.gz")
    endif()
endif()

if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM)
    # for publish
    set(INFER_LITE_PUBLISH_ROOT "${CMAKE_BINARY_DIR}/inference_lite_lib.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}")
    if (LITE_WITH_OPENCL) 
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.opencl")
    endif(LITE_WITH_OPENCL)
    message(STATUS "publish inference lib to ${INFER_LITE_PUBLISH_ROOT}")

    # The final target for publish lite lib
    add_custom_target(publish_inference_lite)
    
    # add cxx lib
    add_custom_target(publish_inference_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${CMAKE_BINARY_DIR}/paddle/fluid/lite/api/model_optimize_tool" "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND cp "${CMAKE_BINARY_DIR}/paddle/fluid/lite/gen_code/paddle_code_generator" "${INFER_LITE_PUBLISH_ROOT}/bin"
            )
    add_dependencies(publish_inference_cxx_lib model_optimize_tool)
    add_dependencies(publish_inference_cxx_lib paddle_code_generator)
    add_dependencies(publish_inference_cxx_lib bundle_full_api)
    add_dependencies(publish_inference_cxx_lib bundle_light_api)
    add_dependencies(publish_inference_lite publish_inference_cxx_lib)

    if (LITE_WITH_JAVA)
    # add java lib
        add_custom_target(publish_inference_java_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/java/so"
            COMMAND cp "${CMAKE_BINARY_DIR}/paddle/fluid/lite/api/android/jni/native/libpaddle_lite_jni.so" "${INFER_LITE_PUBLISH_ROOT}/java/so"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/api/android/jni/src" "${INFER_LITE_PUBLISH_ROOT}/java"
        )
        add_dependencies(publish_inference_java_lib paddle_lite_jni)
        add_dependencies(publish_inference_lite publish_inference_java_lib)
    endif()
    
    if ((ARM_TARGET_OS STREQUAL "android") AND (NOT LITE_WITH_OPENCL) AND
            ((ARM_TARGET_ARCH_ABI STREQUAL armv7) OR (ARM_TARGET_ARCH_ABI STREQUAL armv8)))
        # copy
        add_custom_target(publish_inference_android_cxx_demos ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/glog" "${INFER_LITE_PUBLISH_ROOT}/third_party"
            COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/gflags" "${INFER_LITE_PUBLISH_ROOT}/third_party"
            COMMAND cp "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/cxx/Makefile.def" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/cxx/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/cxx/mobile_full" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/cxx/makefiles/mobile_full/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_full/Makefile"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/cxx/mobile_light" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/cxx/makefiles/mobile_light/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_light/Makefile"
        )
        add_dependencies(publish_inference_android_cxx_demos glog gflags)
        add_dependencies(publish_inference_cxx_lib publish_inference_android_cxx_demos)
    
        if (LITE_WITH_JAVA)
            # copy java mobile_light demo/lib
            add_custom_target(publish_inference_android_java_demo ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND cp -r "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/java/android" "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND cp "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/demo/java/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/libs"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm7"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm8"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm64-v8a"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/armeabi-v7a"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/x86"
            )
            add_dependencies(publish_inference_java_lib publish_inference_android_java_demo)
        endif()
    endif()
    
    if (LITE_WITH_OPENCL) 
        add_custom_target(publish_inference_opencl ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/opencl"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl/cl_kernel" "${INFER_LITE_PUBLISH_ROOT}/opencl"
        )
        add_dependencies(publish_inference_cxx_lib publish_inference_opencl)
    endif()
endif()
