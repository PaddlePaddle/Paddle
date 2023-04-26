include(ExternalProject)
set(JSON_PATH
    "${THIRD_PARTY_PATH}/json"
    CACHE STRING "A path setting for external_json path.")
set(JSON_PREFIX_DIR ${JSON_PATH})
set(JSON_REPOSITORY ${GIT_URL}/nlohmann/json.git)
set(JSON_TAG v3.11.2)

set(JSON_INCLUDE_DIR ${JSON_PREFIX_DIR}/src/extern_json/include)
message("JSON_INCLUDE_DIR is ${JSON_INCLUDE_DIR}")
include_directories(${JSON_INCLUDE_DIR})

ExternalProject_Add(
  extern_json
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${JSON_REPOSITORY}
  GIT_TAG ${JSON_TAG}
  PREFIX ${JSON_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")
message("ADD External Project EXTERN JSON")
add_library(json INTERFACE)
add_dependencies(json extern_json)
