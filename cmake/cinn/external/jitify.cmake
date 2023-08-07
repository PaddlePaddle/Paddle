if(NOT WITH_GPU)
  set(JITIFY_FOUND OFF)
  return()
endif()

include(ExternalProject)

# clone jitify to Paddle/third_party
set(JITIFY_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/jitify)
set(JITIFY_URL https://github.com/NVIDIA/jitify.git)
set(JITIFY_TAG 57de649139c866eb83acacfe50c92ad7c6278776)

if(NOT EXISTS ${JITIFY_SOURCE_DIR})
  execute_process(COMMAND ${GIT_EXECUTABLE} clone ${JITIFY_URL}
                          ${JITIFY_SOURCE_DIR})
  execute_process(COMMAND ${GIT_EXECUTABLE} -C ${JITIFY_SOURCE_DIR} checkout -q
                          ${JITIFY_TAG})
else()
  # check git tag
  execute_process(
    COMMAND ${GIT_EXECUTABLE} -C ${JITIFY_SOURCE_DIR} describe --tags
    OUTPUT_VARIABLE CURRENT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${CURRENT_TAG} STREQUAL ${JITIFY_TAG})
    message(STATUS "Checkout JITIFY to ${JITIFY_TAG}")
    execute_process(COMMAND ${GIT_EXECUTABLE} -C ${JITIFY_SOURCE_DIR} checkout
                            -q ${JITIFY_TAG})
  endif()
endif()

ExternalProject_Add(
  external_jitify
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${JITIFY_SOURCE_DIR}
  PREFIX ${THIRD_PARTY_PATH}/jitify
  CONFIGURE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND "")

include_directories(${JITIFY_SOURCE_DIR})

add_library(extern_jitify INTERFACE)
add_dependencies(extern_jitify external_jitify)
set(jitify_deps extern_jitify)
