# Get the latest git tag.
set(CINN_VERSION $ENV{CINN_VERSION})
set(tmp_version "HEAD")
set(TAG_VERSION_REGEX "[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?")
set(COMMIT_VERSION_REGEX "[0-9a-f]+[0-9a-f]+[0-9a-f]+[0-9a-f]+[0-9a-f]+")
while("${CINN_VERSION}" STREQUAL "")
  # Check current branch name
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref ${tmp_version}
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH_NAME
    RESULT_VARIABLE GIT_BRANCH_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${GIT_BRANCH_RESULT})
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0 --always
              ${tmp_version}
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_TAG_NAME
      RESULT_VARIABLE GIT_RESULT
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT ${GIT_RESULT})
      # Check if current branch is release branch
      if(${GIT_BRANCH_NAME} MATCHES "release/${TAG_VERSION_REGEX}")
        # Check the tag is a correct version
        if(${GIT_TAG_NAME} MATCHES "${COMMIT_VERSION_REGEX}")
          # if no tag was found, set CINN_VERSION to 0.0.0 to represent latest
          set(CINN_VERSION "0.0.0")
        elseif(${GIT_TAG_NAME} MATCHES "v${TAG_VERSION_REGEX}")
          string(REPLACE "v" "" CINN_VERSION ${GIT_TAG_NAME})
        else() # otherwise, get the previous git tag name.
          set(tmp_version "${GIT_TAG_NAME}~1")
        endif()
      else()
        execute_process(
          COMMAND ${GIT_EXECUTABLE} describe --exact-match --tags ${tmp_version}
          WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
          OUTPUT_VARIABLE GIT_EXACT_TAG_NAME
          RESULT_VARIABLE GIT_EXACT_TAG_RESULT
          ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT ${GIT_EXACT_TAG_NAME})
          # Check if current branch is tag branch
          if(${GIT_EXACT_TAG_NAME} MATCHES "v${TAG_VERSION_REGEX}")
            string(REPLACE "v" "" CINN_VERSION ${GIT_EXACT_TAG_NAME})
          else()
            set(CINN_VERSION "0.0.0")
          endif()
        else()
          # otherwise, we always set CINN_VERSION to 0.0.0 to represent latest
          set(CINN_VERSION "0.0.0")
        endif()
      endif()
    else()
      set(CINN_VERSION "0.0.0")
      message(WARNING "Cannot add CINN version from git tag")
    endif()
  else()
    set(CINN_VERSION "0.0.0")
    message(WARNING "Cannot add CINN version for wrong git branch result")
  endif()
endwhile()

string(REPLACE "-" "." CINN_VER_LIST ${CINN_VERSION})
string(REPLACE "." ";" CINN_VER_LIST ${CINN_VER_LIST})
list(GET CINN_VER_LIST 0 CINN_MAJOR_VER)
list(GET CINN_VER_LIST 1 CINN_MINOR_VER)
list(GET CINN_VER_LIST 2 CINN_PATCH_VER)
math(EXPR CINN_VERSION_INTEGER "${CINN_MAJOR_VER} * 1000000
    + ${CINN_MINOR_VER} * 1000 + ${CINN_PATCH_VER}")

add_definitions(-DCINN_VERSION=${CINN_VERSION})
add_definitions(-DCINN_VERSION_INTEGER=${CINN_VERSION_INTEGER})
message(
  STATUS
    "CINN version is ${CINN_VERSION} (major: ${CINN_MAJOR_VER}, minor: ${CINN_MINOR_VER}, patch: ${CINN_PATCH_VER})"
)
