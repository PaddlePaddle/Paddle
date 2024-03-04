if(NOT WITH_GPU)
  set(JITIFY_FOUND OFF)
  return()
endif()

include(ExternalProject)

# clone jitify to Paddle/third_party
set(JITIFY_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/jitify)
set(JITIFY_URL ${GIT_URL}/NVIDIA/jitify.git)
set(JITIFY_TAG 57de649139c866eb83acacfe50c92ad7c6278776)

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
