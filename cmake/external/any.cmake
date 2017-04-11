INCLUDE(ExternalProject)

SET(ANY_SOURCE_DIR ${THIRD_PARTY_PATH}/any)

INCLUDE_DIRECTORIES(${ANY_SOURCE_DIR}/src/linb_any)

ExternalProject_Add(
    linb_any
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/thelink2012/any.git"
    GIT_TAG         "8fef1e93710a0edf8d7658999e284a1142c4c020"
    PREFIX          ${ANY_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

add_definitions(-DANY_IMPL_ANY_CAST_MOVEABLE)
