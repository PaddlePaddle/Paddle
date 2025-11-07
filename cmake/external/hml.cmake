include(ExternalProject)

set(HML_INSTALL_DIR ${THIRD_PARTY_PATH}/install/hml)
set(HML_INC_DIR ${HML_INSTALL_DIR}/include)
set(HML_LIB_DIR ${HML_INSTALL_DIR}/lib)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${HML_LIB_DIR}")
set(HML_DOWNLOAD_DIR ${PADDLE_SOURCE_DIR}/third_party/hml/${CMAKE_SYSTEM_NAME})

set(HML_FILE
    "hml.tgz"
    CACHE STRING "" FORCE)
set(HML_URL
    "https://download.sourcefind.cn:65024/directlink/8/c86-devkit/previous-release/hml/hml-1.6.2-part/${HML_FILE}"
    CACHE STRING "" FORCE)
set(HML_URL_MD5 "a7f855d973cde3e125ca31937b61c7c9")
set(HML_LIB ${HML_LIB_DIR}/libhml_rt.so)
set(HML_SHARED_LIB ${HML_LIB_DIR}/libhml_rt.so)

set(HML_PROJECT "extern_hml")
message(STATUS "HML_FILE: ${HML_FILE}, HML_URL: ${HML_URL}")
set(HML_PREFIX_DIR ${THIRD_PARTY_PATH}/hml)

function(download_hml)
  message(STATUS "Downloading ${HML_URL} to ${HML_DOWNLOAD_DIR}/${HML_FILE}")
  file(
    DOWNLOAD ${HML_URL} ${HML_DOWNLOAD_DIR}/${HML_FILE}
    EXPECTED_MD5 ${HML_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${HML_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${HML_FILE} again"
    )
  endif()
endfunction()

if(EXISTS ${HML_DOWNLOAD_DIR}/${HML_FILE})
  file(MD5 ${HML_DOWNLOAD_DIR}/${HML_FILE} HML_MD5)
  if(NOT HML_MD5 STREQUAL HML_URL_MD5)
    file(REMOVE_RECURSE ${HML_PREFIX_DIR})
    file(REMOVE_RECURSE ${HML_INSTALL_DIR})
    download_hml()
  endif()
else()
  download_hml()
endif()

ExternalProject_Add(
  ${HML_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${HML_DOWNLOAD_DIR}/${HML_FILE}
  URL_MD5 ${HML_URL_MD5}
  DOWNLOAD_DIR ${HML_DOWNLOAD_DIR}
  SOURCE_DIR ${HML_INSTALL_DIR}
  PREFIX ${HML_PREFIX_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${HML_LIB})

include_directories(${HML_INC_DIR})

add_library(hml SHARED IMPORTED GLOBAL)
set_property(TARGET hml PROPERTY IMPORTED_LOCATION ${HML_LIB})
add_dependencies(hml ${HML_PROJECT})
