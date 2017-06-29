INCLUDE(ExternalProject)

SET(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3)

INCLUDE_DIRECTORIES(${EIGEN_SOURCE_DIR}/src/eigen3)

ExternalProject_Add(
    eigen3
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL            "https://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
    URL_MD5        "1a47e78efe365a97de0c022d127607c3"
    PREFIX          ${EIGEN_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

LIST(APPEND external_project_dependencies eigen3)
