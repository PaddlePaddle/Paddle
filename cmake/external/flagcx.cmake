set(CMAKE_FIND_DEBUG_MODE ON)
# flagcx.cmake
if(NOT WITH_FLAGCX)
  return()
endif()

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
