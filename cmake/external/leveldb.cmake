INCLUDE(ExternalProject)

SET(LEVELDB_CXXFLAGS "-fPIC")

# we need to add D_GLIBCXX_USE_CXX11_ABI=0 in ascend when with paddle
IF (WITH_HIERARCHICAL_HCCL)
SET(LEVELDB_CXXFLAGS "${LEVELDB_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
MESSAGE(STATUS "Build in ascend platform: add -D_GLIBCXX_USE_CXX11_ABI=0 for leveldb")
ENDIF()

SET(LEVELDB_SOURCES_DIR ${THIRD_PARTY_PATH}/leveldb)
SET(LEVELDB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/leveldb)
SET(LEVELDB_INCLUDE_DIR "${LEVELDB_INSTALL_DIR}/include" CACHE PATH "leveldb include directory." FORCE)
SET(LEVELDB_LIBRARIES "${LEVELDB_INSTALL_DIR}/lib/libleveldb.a" CACHE FILEPATH "leveldb library." FORCE)
INCLUDE_DIRECTORIES(${LEVELDB_INCLUDE_DIR})

ExternalProject_Add(
        extern_leveldb
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX ${LEVELDB_SOURCES_DIR}
        GIT_REPOSITORY "https://github.com/google/leveldb"
        GIT_TAG v1.18
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND export CXXFLAGS=${LEVELDB_CXXFLAGS} && make -j ${NUM_OF_PROCESSOR} libleveldb.a
        INSTALL_COMMAND mkdir -p ${LEVELDB_INSTALL_DIR}/lib/
        && cp ${LEVELDB_SOURCES_DIR}/src/extern_leveldb/libleveldb.a ${LEVELDB_LIBRARIES}
        && cp -r ${LEVELDB_SOURCES_DIR}/src/extern_leveldb/include ${LEVELDB_INSTALL_DIR}/
        BUILD_IN_SOURCE 1
)

ADD_DEPENDENCIES(extern_leveldb snappy)

ADD_LIBRARY(leveldb STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET leveldb PROPERTY IMPORTED_LOCATION ${LEVELDB_LIBRARIES})
ADD_DEPENDENCIES(leveldb extern_leveldb)

LIST(APPEND external_project_dependencies leveldb)
