INCLUDE(ExternalProject)

set(XXHASH_SOURCE_DIR ${THIRD_PARTY_PATH}/xxhash)
set(XXHASH_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xxhash)
set(XXHASH_INCLUDE_DIR "${XXHASH_INSTALL_DIR}/include")

IF(WITH_STATIC_LIB)
  SET(BUILD_CMD make lib)
ELSE()
  IF(APPLE)
    SET(BUILD_CMD sed -i \"\" "s/-Wstrict-prototypes -Wundef/-Wstrict-prototypes -Wundef -fPIC/g" ${XXHASH_SOURCE_DIR}/src/extern_xxhash/Makefile && make lib)
  ELSE(APPLE)
    SET(BUILD_CMD sed -i "s/-Wstrict-prototypes -Wundef/-Wstrict-prototypes -Wundef -fPIC/g" ${XXHASH_SOURCE_DIR}/src/extern_xxhash/Makefile && make lib)
  ENDIF(APPLE)
ENDIF()

if(WIN32)
  ExternalProject_Add(
          extern_xxhash
          ${EXTERNAL_PROJECT_LOG_ARGS}
          GIT_REPOSITORY  "https://github.com/Cyan4973/xxHash"
          GIT_TAG         "v0.6.5"
          PREFIX          ${XXHASH_SOURCE_DIR}
          DOWNLOAD_NAME   "xxhash"
          UPDATE_COMMAND  ""
          BUILD_IN_SOURCE 1
          PATCH_COMMAND
          CONFIGURE_COMMAND
          ${CMAKE_COMMAND} ${XXHASH_SOURCE_DIR}/src/extern_xxhash/cmake_unofficial
          -DCMAKE_INSTALL_PREFIX:PATH=${XXHASH_INSTALL_DIR}
          -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
          -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
          -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
          -DBUILD_XXHSUM=OFF
          -DCMAKE_GENERATOR_PLATFORM=x64
          -DBUILD_SHARED_LIBS=OFF
          ${OPTIONAL_CACHE_ARGS}
          TEST_COMMAND      ""
  )
else()
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
endif()

if (WIN32)
  IF(NOT EXISTS "${XXHASH_INSTALL_DIR}/lib/libxxhash.lib")
    add_custom_command(TARGET extern_xxhash POST_BUILD
            COMMAND cmake -E copy ${XXHASH_INSTALL_DIR}/lib/xxhash.lib ${XXHASH_INSTALL_DIR}/lib/libxxhash.lib
            )
  ENDIF()
  set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/libxxhash.lib")
else()
  set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/libxxhash.a")
endif ()
INCLUDE_DIRECTORIES(${XXHASH_INCLUDE_DIR})

add_library(xxhash STATIC IMPORTED GLOBAL)
set_property(TARGET xxhash PROPERTY IMPORTED_LOCATION ${XXHASH_LIBRARIES})
include_directories(${XXHASH_INCLUDE_DIR})
add_dependencies(xxhash extern_xxhash)
