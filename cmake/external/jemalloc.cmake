include(ExternalProject)

if (NOT UNIX AND NOT LINUX AND NOT APPLE)
  MESSAGE(WARNING, "Jemalloc can not be built automaticlly, please build it manually and put it at " ${THIRD_PARTY_PATH}/install/jemalloc)
  MESSAGE(WARNING, "Please refer to https://github.com/jemalloc/jemalloc.git to build jemalloc in your OS")
endif()

set(JEMALLOC_SOURCES_DIR ${THIRD_PARTY_PATH}/jemalloc)
set(JEMALLOC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/jemalloc)
set(JEMALLOC_INCLUDE_DIR "${JEMALLOC_INSTALL_DIR}/include" CACHE PATH "jemalloc include directory." FORCE)

if(WIN32)
  set(JEMALLOC_LIBRARIES "${JEMALLOC_INSTALL_DIR}/lib/libjemalloc.lib" CACHE PATH "jemalloc library." FORCE)
else()
  set(JEMALLOC_LIBRARIES "${JEMALLOC_INSTALL_DIR}/lib/libjemalloc.a" CACHE PATH "jemalloc library" FORCE)
endif()

include_directories(${JEMALLOC_INCLUDE_DIR})

if(UNIX OR LINUX OR APPLE)
  # Jemalloc needs autoconf to build, so we should install autoconf first
  execute_process(COMMAND which autoconf RESULT_VARIABLE AUTOCONF_NOT_EXIST OUTPUT_QUIET)
  if (AUTOCONF_NOT_EXIST)
    set(AUTOCONF_VERSION "2.69")

    MESSAGE(STATUS "Autoconf not found, try to install autoconf-${AUTOCONF_VERSION}")
    set(AUTOCONF_TAR "autoconf-${AUTOCONF_VERSION}")
    set(AUTOCONF_URL "http://ftp.gnu.org/gnu/autoconf/${AUTOCONF_TAR}.tar.gz")
    set(AUTOCONF_DOWNLOAD_DIR ${THIRD_PARTY_PATH}/autoconf/extern_autoconf)

    set(AUTOCONF_BUILD_COMMAND "wget ${AUTOCONF_URL} -c -q -O ${AUTOCONF_DOWNLOAD_DIR}/${AUTOCONF_TAR}.tar.gz \
        && cd ${AUTOCONF_DOWNLOAD_DIR} \
        && tar zxf ${AUTOCONF_TAR}.tar.gz \
        && cd ${AUTOCONF_TAR} \
        && ./configure && make && make install")
    set(AUTOCONF_BUILD_FILE ${AUTOCONF_DOWNLOAD_DIR}/autoconf_build.sh)
    file(WRITE ${AUTOCONF_BUILD_FILE} ${AUTOCONF_BUILD_COMMAND}) 

    execute_process(COMMAND mkdir -p ${AUTOCONF_DOWNLOAD_DIR})
    execute_process(COMMAND sh ${AUTOCONF_BUILD_FILE} RESULT_VARIABLE AUTOCONF_INSTALL_FAILURE OUTPUT_QUIET ERROR_QUIET)
    if (AUTOCONF_INSTALL_FAILURE)
      MESSAGE(FATAL_ERROR "Cannot install autoconf, which is required to build jemalloc.
              In Ubuntu, you can use 'apt-get install -y autoconf' to install it.")
    endif()
  endif()

  execute_process(COMMAND which autoconf OUTPUT_VARIABLE AUTOCONF_PATH)
  string(STRIP ${AUTOCONF_PATH} AUTOCONF_PATH)
  MESSAGE(STATUS "Autoconf: ${AUTOCONF_PATH}")

  set(JEMALLOC_BUILD_COMMAND cd ${JEMALLOC_SOURCES_DIR}/src/extern_jemalloc
      && ./autogen.sh --with-jemalloc-prefix=je_ --prefix ${JEMALLOC_INSTALL_DIR} && make -j8)
  set(JEMALLOC_INSTALL_COMMAND cd ${JEMALLOC_SOURCES_DIR}/src/extern_jemalloc
      && make install_bin install_include install_lib)

  ExternalProject_Add(
    extern_jemalloc
    GIT_REPOSITORY    "https://github.com/jemalloc/jemalloc.git"
    GIT_TAG           "5.1.0"
    PREFIX            ${JEMALLOC_SOURCES_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ${JEMALLOC_BUILD_COMMAND}
    INSTALL_COMMAND   ${JEMALLOC_INSTALL_COMMAND}
    UPDATE_COMMAND    ""
  )
endif()

add_library(jemalloc STATIC IMPORTED GLOBAL)
set_property(TARGET jemalloc PROPERTY IMPORTED_LOCATION ${JEMALLOC_LIBRARIES})
add_dependencies(jemalloc extern_jemalloc)
LIST(APPEND external_project_dependencies jemalloc)
