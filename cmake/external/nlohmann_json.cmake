INCLUDE(ExternalProject)

SET(NLOHMANN_JSON_SOURCE_DIR ${THIRD_PARTY_PATH}/nlohmann_json)

INCLUDE_DIRECTORIES(${NLOHMANN_JSON_SOURCE_DIR}/src/nlohmann_json/src/)

ExternalProject_Add(
    nlohmann_json
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/nlohmann/json.git"
    GIT_TAG         "v2.1.1"
    PREFIX          ${NLOHMANN_JSON_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

LIST(APPEND external_project_dependencies nlohmann_json)
