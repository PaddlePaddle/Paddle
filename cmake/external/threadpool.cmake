INCLUDE(ExternalProject)

SET(THREADPOOL_SOURCE_DIR ${THIRD_PARTY_PATH}/threadpool)
SET(THREADPOOL_INCLUDE_DIR ${THREADPOOL_SOURCE_DIR}/src/extern_threadpool)
INCLUDE_DIRECTORIES(${THREADPOOL_INCLUDE_DIR})

ExternalProject_Add(
    extern_threadpool
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/progschj/ThreadPool.git"
    GIT_TAG         9a42ec1329f259a5f4881a291db1dcb8f2ad9040
    PREFIX          ${THREADPOOL_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/threadpool_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_threadpool = \"${dummyfile}\";")
    add_library(simple_threadpool STATIC ${dummyfile})
else()
    add_library(simple_threadpool INTERFACE)
endif()

add_dependencies(simple_threadpool extern_threadpool)

LIST(APPEND external_project_dependencies simple_threadpool)
