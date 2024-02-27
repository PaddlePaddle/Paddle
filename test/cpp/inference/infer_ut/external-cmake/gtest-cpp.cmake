find_package(Git REQUIRED)
message("${CMAKE_BUILD_TYPE}")
set(GTEST_PREFIX_DIR ${CMAKE_CURRENT_BINARY_DIR}/gtest)
set(PADDLE_SOURCE_DIR $ENV{PADDLE_SOURCE_DIR})
set(GTEST_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/gtest)
set(GTEST_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install/gtest)
set(GTEST_INCLUDE_DIR
    "${GTEST_INSTALL_DIR}/include"
    CACHE PATH "gtest include directory." FORCE)
set(GTEST_TAG release-1.8.1)
include_directories(${GTEST_INCLUDE_DIR})
if(WIN32)
  # if use CMAKE_INSTALL_LIBDIR, the path of lib actually is \
  # install/gtest/lib/gtest.lib but GTEST_LIBRARIES
  # is install/gtest/gtest.lib
  set(GTEST_LIBRARIES
      "${GTEST_INSTALL_DIR}/lib/gtest.lib"
      CACHE FILEPATH "gtest libraries." FORCE)
  set(GTEST_MAIN_LIBRARIES
      "${GTEST_INSTALL_DIR}/lib/gtest_main.lib"
      CACHE FILEPATH "gtest main libraries." FORCE)
else()
  set(GTEST_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgtest.a"
      CACHE FILEPATH "gtest libraries." FORCE)
  set(GTEST_MAIN_LIBRARIES
      "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgtest_main.a"
      CACHE FILEPATH "gtest main libraries." FORCE)
endif()
ExternalProject_Add(
  extern_gtest
  PREFIX gtest
  SOURCE_DIR ${GTEST_SOURCE_DIR}
  DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR}
             -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
             -DCMAKE_BUILD_TYPE:STRING=Release
  BUILD_BYPRODUCTS ${GTEST_LIBRARIES}
  BUILD_BYPRODUCTS ${GTEST_MAIN_LIBRARIES})

add_library(thirdparty_gtest STATIC IMPORTED GLOBAL)
set_property(TARGET thirdparty_gtest PROPERTY IMPORTED_LOCATION
                                              ${GTEST_LIBRARIES})
add_dependencies(thirdparty_gtest extern_gtest)

add_library(thirdparty_gtest_main STATIC IMPORTED GLOBAL)
set_property(TARGET thirdparty_gtest_main PROPERTY IMPORTED_LOCATION
                                                   ${GTEST_MAIN_LIBRARIES})
add_dependencies(thirdparty_gtest_main extern_gtest)
