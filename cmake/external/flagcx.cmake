set(CMAKE_FIND_DEBUG_MODE ON)
# flagcx.cmake
if(NOT WITH_FLAGCX)
  return()
endif()

if(WITH_XPU)

  #Paths
  set(FLAGCX_SOURCE_DIR "${PADDLE_SOURCE_DIR}/third_party/flagcx")
  set(FLAGCX_PREFIX "${FLAGCX_BINARY_DIR}") # staged "install"
  set(FLAGCX_INC_SRC "${FLAGCX_SOURCE_DIR}/flagcx/include") # headers in source
  set(FLAGCX_LIB_NAME
      "flagcx"
      CACHE STRING "FlagCX library base name")
  set(FLAGCX_LIB "${FLAGCX_SOURCE_DIR}/build/lib/libflagcx.so")
  set(XPU_INCLUDE_PATH "${THIRD_PARTY_PATH}/install/xpu/include/xpu")
  set(XPU_LIB_PATH "${THIRD_PARTY_PATH}/install/xpu/lib")

  find_path(
    FLAGCX_INCLUDE_DIR flagcx.h
    PATHS ${FLAGCX_SOURCE_DIR}/flagcx/include
    NO_DEFAULT_PATH)
  message(STATUS "FLAGCX_INCLUDE_DIR is ${FLAGCX_INCLUDE_DIR}")
  include_directories(SYSTEM ${FLAGCX_INCLUDE_DIR})

  ExternalProject_Add(
    flagcx_ep
    SOURCE_DIR "${FLAGCX_SOURCE_DIR}"
    BINARY_DIR "${FLAGCX_SOURCE_DIR}"
    CONFIGURE_COMMAND "" # none
    # Ensure the script is executable
    BUILD_COMMAND bash ${CMAKE_SOURCE_DIR}/tools/flagcx/build_flagcx_xpu.sh
                  ${XPU_INCLUDE_PATH} ${XPU_LIB_PATH} ${FLAGCX_SOURCE_DIR}
    # Option A: let the script do the staging; then INSTALL_COMMAND is empty
    INSTALL_COMMAND ""
    LOG_BUILD 1
    LOG_INSTALL 1)

  add_library(flagcx INTERFACE)
  add_dependencies(flagcx flagcx_ep)
else()

  set(FLAGCX_SOURCE_DIR "${PADDLE_SOURCE_DIR}/third_party/flagcx")
  set(FLAGCX_BINARY_DIR "${PADDLE_SOURCE_DIR}/build/third_party/flagcx")
  set(THIRD_PARTY_DIR "${PADDLE_SOURCE_DIR}/build/third_party")
  set(FLAGCX_ROOT "/usr/local/flagcx")
  set(FLAGCX_LIB_DIR "${FLAGCX_BINARY_DIR}/build/lib")
  set(USR_LOCAL_DIR "/usr/local")

  file(REMOVE_RECURSE ${FLAGCX_BINARY_DIR})
  message(STATUS "removed old flagcx dir")
  message(STATUS "Copying third-party source to build directory")
  execute_process(COMMAND cp -r ${FLAGCX_SOURCE_DIR} ${THIRD_PARTY_DIR}
                  RESULT_VARIABLE COPY_RESULT)

  if(NOT COPY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to copy third-party source to build directory")
  endif()

  # Create a custom target to build the third-party library
  message(STATUS "Building third-party library with its Makefile")
  execute_process(
    COMMAND make
    WORKING_DIRECTORY ${FLAGCX_BINARY_DIR}
    RESULT_VARIABLE BUILD_RESULT)

  find_path(
    FLAGCX_INCLUDE_DIR flagcx.h
    PATHS ${FLAGCX_SOURCE_DIR}/flagcx/include
    NO_DEFAULT_PATH)

  message(STATUS "FLAGCX_INCLUDE_DIR is ${FLAGCX_INCLUDE_DIR}")
  include_directories(SYSTEM ${FLAGCX_INCLUDE_DIR})

  add_library(flagcx INTERFACE)
  find_library(
    FLAGCX_LIB
    NAMES flagcx libflagcx
    PATHS ${FLAGCX_LIB_DIR}
    DOC "My custom library")

  add_dependencies(flagcx FLAGCX_LIB)
  message(STATUS "FLAGCX_LIB is ${FLAGCX_LIB}")
endif()
