# Get the latest git tag.
set(PADDLE_VERSION $ENV{PADDLE_VERSION})
set(tmp_version "HEAD")
while ("${PADDLE_VERSION}" STREQUAL "")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0 ${tmp_version}
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_TAG_NAME
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT ${GIT_RESULT})
    # Check the tag is a correct version
    if (${GIT_TAG_NAME} MATCHES "v[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?")
      string(REPLACE "v" "" PADDLE_VERSION ${GIT_TAG_NAME})
    else()  # otherwise, get the previous git tag name.
      set(tmp_version "${GIT_TAG_NAME}~1")
    endif()
  else()
    set(PADDLE_VERSION "0.0.0")
    message(WARNING "Cannot add paddle version from git tag")
  endif()
endwhile()

add_definitions(-DPADDLE_VERSION=${PADDLE_VERSION})
message(STATUS "Paddle version is ${PADDLE_VERSION}")
