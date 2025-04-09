# Get the latest git tag.
set(PADDLE_VERSION $ENV{PADDLE_VERSION})
if(WITH_NIGHTLY_BUILD)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} show -s --format=%ci HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_TIME
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE " (.*)$" "" DATE_ONLY "${GIT_COMMIT_TIME}")
  string(REPLACE "-" "" DATE_ONLY "${DATE_ONLY}")
  # Print the last commit date
  message(STATUS "Last commit date: ${DATE_ONLY}")
  set(PADDLE_VERSION "${PADDLE_VERSION}.dev${DATE_ONLY}")
endif()
set(tmp_version "HEAD")
set(TAG_VERSION_REGEX "[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?")
set(COMMIT_VERSION_REGEX "[0-9a-f]+[0-9a-f]+[0-9a-f]+[0-9a-f]+[0-9a-f]+")
while("${PADDLE_VERSION}" STREQUAL "")
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
          # if no tag was found, set PADDLE_VERSION to 0.0.0 to represent latest
          set(PADDLE_VERSION "0.0.0")
        elseif(${GIT_TAG_NAME} MATCHES "v${TAG_VERSION_REGEX}")
          string(REPLACE "v" "" PADDLE_VERSION ${GIT_TAG_NAME})
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
            string(REPLACE "v" "" PADDLE_VERSION ${GIT_EXACT_TAG_NAME})
          else()
            set(PADDLE_VERSION "0.0.0")
          endif()
        else()
          # otherwise, we always set PADDLE_VERSION to 0.0.0 to represent latest
          set(PADDLE_VERSION "0.0.0")
        endif()
      endif()
    else()
      set(PADDLE_VERSION "0.0.0")
      message(WARNING "Cannot add paddle version from git tag")
    endif()
  else()
    set(PADDLE_VERSION "0.0.0")
    message(WARNING "Cannot add paddle version for wrong git branch result")
  endif()
endwhile()

string(REPLACE "-" "." PADDLE_VER_LIST ${PADDLE_VERSION})
string(REPLACE "." ";" PADDLE_VER_LIST ${PADDLE_VER_LIST})
list(GET PADDLE_VER_LIST 0 PADDLE_MAJOR_VER)
list(GET PADDLE_VER_LIST 1 PADDLE_MINOR_VER)
list(GET PADDLE_VER_LIST 2 PADDLE_PATCH_VER)

math(EXPR PADDLE_VERSION_INTEGER "${PADDLE_MAJOR_VER} * 1000000
    + ${PADDLE_MINOR_VER} * 1000 + ${PADDLE_PATCH_VER}")

add_definitions(-DPADDLE_VERSION=${PADDLE_VERSION})
add_definitions(-DPADDLE_VERSION_INTEGER=${PADDLE_VERSION_INTEGER})
message(STATUS "Paddle version is ${PADDLE_VERSION}")

# write paddle version
function(version version_file)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -1
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
    OUTPUT_VARIABLE PADDLE_GIT_COMMIT)
  file(
    WRITE ${version_file}
    "Paddle version: ${PADDLE_VERSION}\n"
    "GIT COMMIT ID: ${PADDLE_GIT_COMMIT}\n"
    "WITH_MKL: ${WITH_MKL}\n"
    "WITH_ONEDNN: ${WITH_ONEDNN}\n"
    "WITH_OPENVINO: ${WITH_OPENVINO}\n"
    "WITH_GPU: ${WITH_GPU}\n"
    "WITH_ROCM: ${WITH_ROCM}\n"
    "WITH_IPU: ${WITH_IPU}\n")
  if(WITH_GPU)
    file(APPEND ${version_file}
         "CUDA version: ${CUDA_VERSION}\n"
         "CUDNN version: v${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}\n")
  endif()
  if(WITH_ROCM)
    file(APPEND ${version_file}
         "HIP version: v${HIP_MAJOR_VERSION}.${HIP_MINOR_VERSION}\n"
         "MIOpen version: v${MIOPEN_MAJOR_VERSION}.${MIOPEN_MINOR_VERSION}\n")
  endif()
  if(WITH_IPU)
    file(APPEND ${version_file} "PopART version: ${POPART_VERSION}\n")
  endif()
  file(APPEND ${version_file}
       "CXX compiler version: ${CMAKE_CXX_COMPILER_VERSION}\n")
  if(TENSORRT_FOUND)
    file(
      APPEND ${version_file}
      "WITH_TENSORRT: ${TENSORRT_FOUND}\n"
      "TensorRT version: v${TENSORRT_MAJOR_VERSION}.${TENSORRT_MINOR_VERSION}.${TENSORRT_PATCH_VERSION}.${TENSORRT_BUILD_VERSION}\n"
    )
  endif()
endfunction()
