include(ExternalProject)

set(ROBIN_MAP_SOURCE_DIR ${THIRD_PARTY_PATH}/robin_map)
set(ROBIN_MAP_INCLUDE_DIR ${ROBIN_MAP_SOURCE_DIR}/src/extern_robin_map/include)

include_directories(${ROBIN_MAP_INCLUDE_DIR})

ExternalProject_Add(
  extern_robin_map
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY "https://github.com/Tessil/robin-map.git"
  GIT_TAG        "v0.5.0"
  PREFIX         ${ROBIN_MAP_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/robin_map_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(robin_map STATIC ${dummyfile})
else()
  add_library(robin_map INTERFACE)
endif()

add_dependencies(robin_map extern_robin_map)

LIST(APPEND externl_project_dependencies robin_map)
