INCLUDE(ExternalProject)

set(XXHASH_SOURCE_DIR ${THIRD_PARTY_PATH}/xxhash)
set(XXHASH_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xxhash)
set(XXHASH_INCLUDE_DIR "${XXHASH_INSTALL_DIR}/include")


ExternalProject_Add(
    extern_xxhash
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/Cyan4973/xxHash"
    GIT_TAG         "v0.6.5"
    PREFIX          ${XXHASH_SOURCE_DIR}
    DOWNLOAD_NAME   "xxhash"
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    PATCH_COMMAND
    BUILD_COMMAND     make lib
    INSTALL_COMMAND   export PREFIX=${XXHASH_INSTALL_DIR}/ && make install
    TEST_COMMAND      ""
)


set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/libxxhash.a")
INCLUDE_DIRECTORIES(${XXHASH_INCLUDE_DIR})

add_library(xxhash STATIC IMPORTED GLOBAL)
set_property(TARGET xxhash PROPERTY IMPORTED_LOCATION ${XXHASH_LIBRARIES})
include_directories(${XXHASH_INCLUDE_DIR})
add_dependencies(xxhash extern_xxhash)

