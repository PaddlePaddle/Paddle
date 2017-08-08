INCLUDE(ExternalProject)

SET(ANY_SOURCE_DIR ${THIRD_PARTY_PATH}/any)

INCLUDE_DIRECTORIES(${ANY_SOURCE_DIR}/src/extern_lib_any)

ExternalProject_Add(
    extern_lib_any
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/PaddlePaddle/any.git"
    GIT_TAG         "15595d8324be9e8a9a80d9ae442fdd12bd66df5d"
    PREFIX          ${ANY_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/lib_any_dummy.c)
    file(WRITE ${dummyfile} "const char * dummy_any = \"${dummyfile}\";")
    add_library(lib_any STATIC ${dummyfile})
else()
    add_library(lib_any INTERFACE)
endif()

add_dependencies(lib_any extern_lib_any)

add_definitions(-DANY_IMPL_ANY_CAST_MOVEABLE)
LIST(APPEND external_project_dependencies lib_any)
