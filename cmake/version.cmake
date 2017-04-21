# Get the latest git tag.
set(PADDLE_VERSION $ENV{PADDLE_VERSION})
if ("${PADDLE_VERSION}" STREQUAL "")  # parse from git tags.
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0 --exact "HEAD"
    WORKING_DIRECTORY ${PROJ_ROOT}
    OUTPUT_VARIABLE GIT_TAG_NAME
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (NOT ${GIT_RESULT})
    # Check the tag is a correct version
    if (${GIT_TAG_NAME} MATCHES "v[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?")
      string(REPLACE "v" "" PADDLE_VERSION ${GIT_TAG_NAME})
    endif()
  endif()
endif()

if ("${PADDLE_VERSION}" STREQUAL "") # parse from git sha
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
    WORKING_DIRECTORY ${PROJ_ROOT}
    OUTPUT_VARIABLE GIT_SHA
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (NOT ${GIT_RESULT})
    set(PADDLE_VERSION ${GIT_SHA})
  else()  # cannot parse version.
    set(PADDLE_VERSION 0.0.0)
  endif()
endif()

add_definitions(-DPADDLE_VERSION=${PADDLE_VERSION})
message(STATUS "Paddle version is ${PADDLE_VERSION}")
