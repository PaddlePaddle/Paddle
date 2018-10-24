INCLUDE(ExternalProject)

set(XXHASH_SOURCE_DIR ${THIRD_PARTY_PATH}/xxhash)
set(XXHASH_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xxhash)
set(XXHASH_INCLUDE_DIR "${XXHASH_INSTALL_DIR}/include")


ExternalProject_Add(
    extern_xxhash
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/Cyan4973/xxHash"
    # eigen on cuda9.1 missing header of math_funtions.hpp
    # https://stackoverflow.com/questions/43113508/math-functions-hpp-not-found-when-using-cuda-with-eigen
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
#if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
#    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/lib_xxhash_dummy.c)
#    file(WRITE ${dummyfile} "const char * dummy_any = \"${dummyfile}\";")
#    add_library(lib_xxhash STATIC ${dummyfile})
#else()
#    add_library(lib_xxhash INTERFACE)
#endif()
include_directories(${XXHASH_INCLUDE_DIR})
add_dependencies(xxhash extern_xxhash)
#LIST(APPEND external_project_dependencies xxhash)
#link_libraries(${XXHASH_LIBRARIES})

