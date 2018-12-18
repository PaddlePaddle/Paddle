IF(NOT ${WITH_TBB})
  return()
ENDIF(NOT ${WITH_TBB})

IF(WIN32 OR APPLE)
    MESSAGE(WARNING
        "Windows or Mac is not supported with TBB in Paddle yet."
        "Force WITH_TBB=OFF")
    SET(WITH_TBB OFF CACHE STRING "Disable TBB package in Windows and MacOS" FORCE)
    return()
ENDIF()

SET(TBB_PROJECT "extern_tbb")
SET(TBB_SOURCES_DIR ${THIRD_PARTY_PATH}/tbb)
SET(TBB_INCLUDE_DIR ${THIRD_PARTY_PATH}/tbb/src/extern_tbb/include)
SET(TBB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/tbb)

SET(TBB_FLAG "-Wno-error=strict-overflow -Wno-error=unused-result -Wno-error=array-bounds")
SET(TBB_FLAG "${TBB_FLAG} -Wno-unused-result -Wno-unused-value")
SET(TBB_CFLAG "${CMAKE_C_FLAGS} ${TBB_FLAG}")
SET(TBB_CXXFLAG "${CMAKE_CXX_FLAGS} ${TBB_FLAG}")


INCLUDE(ExternalProject)
ExternalProject_Add(
    ${TBB_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY      "https://github.com/01org/tbb.git"
    # TBB 2019 Update 3
    GIT_TAG             "314792356bf75f4a190277536aea543b9b6b310b"
    PREFIX              ${TBB_SOURCES_DIR}
    UPDATE_COMMAND      ""
    BUILD_IN_SOURCE     1
    # build static library
    BUILD_COMMAND       make -j extra_inc=big_iron.inc
    INSTALL_COMMAND     ""
    CONFIGURE_COMMAND   ""
    DOWNLOAD_DIR        ${TBB_SOURCES_DIR}
)

# tbb library generate in ununified directory with linux dist version/kenrel version as extension.
# search and install the library to third_party/tbb/install
unset(TBB_LIBRARIES_SOURCES)
file(GLOB_RECURSE TBB_LIBRARIES_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${TBB_SOURCES_DIR}/src/extern_tbb/build/*.a")
LIST(REMOVE_DUPLICATES TBB_LIBRARIES_SOURCES)

set(TBB_LIBRARY_NAME "")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(TBB_LIBRARY_NAME "libtbb_debug.a")
else() # RelWithDebInfo Release MinSizeRel
  set(TBB_LIBRARY_NAME "libtbb.a")
endif()

SET(TBB_LIBRARIES_PATH "")
foreach(TBB_LIB ${TBB_LIBRARIES_SOURCES})
  get_filename_component(TBB_LIB_PATH ${TBB_LIB} DIRECTORY)
  LIST(APPEND TBB_LIBRARIES_PATH ${TBB_LIB_PATH})
endforeach()

find_library(TBB_LIBRARIES ${TBB_LIBRARY_NAME} PATHS ${TBB_LIBRARIES_PATH})

add_custom_command(TARGET extern_tbb POST_BUILD
  COMMAND cmake -E make_directory ${TBB_INSTALL_DIR}/lib
  COMMAND cmake -E copy ${TBB_LIBRARIES} ${TBB_INSTALL_DIR}/lib
  COMMAND cmake -E copy_directory ${TBB_INCLUDE_DIR} ${TBB_INSTALL_DIR}/include
  )


if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
  SET(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/tbb_dummy.c)
  FILE(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
  ADD_LIBRARY(tbb STATIC ${dummyfile})
endif()

INCLUDE_DIRECTORIES(${TBB_INSTALL_DIR}/include)
ADD_LIBRARY(tbb STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET tbb PROPERTY IMPORTED_LOCATION
  ${TBB_LIBRARIES})
add_dependencies(tbb extern_tbb)
LIST(APPEND external_project_dependencies tbb)
