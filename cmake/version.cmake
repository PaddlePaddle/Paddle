# Get the latest git tag.
set(PADDLE_VERSION $ENV{PADDLE_VERSION})
set(tmp_version "HEAD")
set(TAG_VERSION_REGEX "[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?")
set(COMMIT_VERSION_REGEX "[0-9a-f]+")
while ("${PADDLE_VERSION}" STREQUAL "")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0 --always ${tmp_version}
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_TAG_NAME
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT ${GIT_RESULT})
    # Check the tag is a correct version
    if (${GIT_TAG_NAME} MATCHES "${COMMIT_VERSION_REGEX}")
      # if no tag was found, set PADDLE_VERSION to latest
      set(PADDLE_VERSION "latest")
    elseif (${GIT_TAG_NAME} MATCHES "v${TAG_VERSION_REGEX}")
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
