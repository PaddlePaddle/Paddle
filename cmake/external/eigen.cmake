INCLUDE(ExternalProject)

SET(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3)

INCLUDE_DIRECTORIES(${EIGEN_SOURCE_DIR}/src/eigen3)

ExternalProject_Add(
    extern_eigen3
    ${EXTERNAL_PROJECT_LOG_ARGS}
    # for latest version, please get from official website
    # URL            "https://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
    # URL_MD5        "1a47e78efe365a97de0c022d127607c3"

    # for no-ssl http support, please get from bazel's mirror
    # URL           "http://mirror.bazel.build/bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz"
    # URL_MD5       "4645c66075982da6fa0bcf6b20f3e8f7"

    # get from github mirror
    GIT_REPOSITORY  "https://github.com/RLovelett/eigen.git"
    GIT_TAG         "a46d2e7337c4656f00abe54a8115f6d76153a048"
    PREFIX          ${EIGEN_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

ADD_LIBRARY(eigen3 INTERFACE)
ADD_DEPENDENCIES(eigen3 extern_eigen3)

LIST(APPEND external_project_dependencies extern_eigen3)
