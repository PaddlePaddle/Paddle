INCLUDE(ExternalProject)

SET(PYBIND_SOURCE_DIR ${THIRD_PARTY_PATH}/pybind)

INCLUDE_DIRECTORIES(${PYBIND_SOURCE_DIR}/src/extern_pybind/include)

ExternalProject_Add(
        extern_pybind
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  "https://github.com/pybind/pybind11.git"
        GIT_TAG         "v2.1.1"
        PREFIX          ${PYBIND_SOURCE_DIR}
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/pybind_dummy.c)
    file(WRITE ${dummyfile} "const char * dummy_any = \"${dummyfile}\";")
    add_library(pybind STATIC ${dummyfile})
else()
    add_library(pybind INTERFACE)
endif()

add_dependencies(pybind extern_pybind)

LIST(APPEND external_project_dependencies pybind)
