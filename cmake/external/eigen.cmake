INCLUDE(ExternalProject)

SET(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3)

INCLUDE_DIRECTORIES(${EIGEN_SOURCE_DIR}/src/)

ExternalProject_Add(
    eigen3
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL            "https://bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz"
    URL_MD5        "4645c66075982da6fa0bcf6b20f3e8f7"
    PREFIX          ${EIGEN_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

LIST(APPEND external_project_dependencies eigen3)