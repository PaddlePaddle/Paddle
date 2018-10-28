INCLUDE(ExternalProject)

set(XXHASH_SOURCE_DIR ${THIRD_PARTY_PATH}/xxhash)
set(XXHASH_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xxhash)
set(XXHASH_INCLUDE_DIR "${XXHASH_INSTALL_DIR}/include")

IF(WITH_STATIC_LIB)
  SET(BUILD_CMD make lib)
ELSE()
  SET(BUILD_CMD sed -i "s/-Wstrict-prototypes -Wundef/-Wstrict-prototypes -Wundef -fPIC/g" ${XXHASH_SOURCE_DIR}/src/extern_xxhash/Makefile && make lib)
ENDIF()

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
    BUILD_COMMAND     ${BUILD_CMD}
    INSTALL_COMMAND   export PREFIX=${XXHASH_INSTALL_DIR}/ && make install
    TEST_COMMAND      ""
)

set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/libxxhash.a")
INCLUDE_DIRECTORIES(${XXHASH_INCLUDE_DIR})

add_library(xxhash STATIC IMPORTED GLOBAL)
set_property(TARGET xxhash PROPERTY IMPORTED_LOCATION ${XXHASH_LIBRARIES})
include_directories(${XXHASH_INCLUDE_DIR})
add_dependencies(xxhash extern_xxhash)

LIST(APPEND external_project_dependencies xxhash)

IF(WITH_C_API)
  INSTALL(DIRECTORY ${XXHASH_INCLUDE_DIR} DESTINATION third_party/xxhash)
  IF(ANDROID)
    INSTALL(FILES ${XXHASH_LIBRARIES} DESTINATION third_party/xxhash/lib/${ANDROID_ABI})
  ELSE()
    INSTALL(FILES ${XXHASH_LIBRARIES} DESTINATION third_party/xxhash/lib)
  ENDIF()
ENDIF()
