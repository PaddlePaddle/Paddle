if(NOT WITH_MUSA)
  return()
endif()

if(NOT DEFINED ENV{MUSA_PATH})
  set(MUSA_PATH
      "/usr/local/musa"
      CACHE PATH "Path to which ROCm has been installed")
else()
  set(MUSA_PATH
      $ENV{MUSA_PATH}
      CACHE PATH "Path to which ROCm has been installed")
endif()
set(CMAKE_MODULE_PATH "${MUSA_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(MUSA REQUIRED)
include_directories(${MUSA_PATH}/include)

# set openmp include directory
set(llvm_openmp_search_list)
foreach(item RANGE 6 20 1)
  list(APPEND llvm_openmp_search_list /usr/lib/llvm-${item}/include/openmp/)
endforeach()

find_path(
  OPENMP_INCLUDE_DIR omp.h
  PATHS ${llvm_openmp_search_list} REQUIRED
  NO_DEFAULT_PATH)
include_directories(${OPENMP_INCLUDE_DIR})

macro(find_musa_version musa_version_file)
  set(python_file ${PROJECT_BINARY_DIR}/get_version.py)
  set(MUSA_VERSION
      "None"
      CACHE STRING "musa version" FORCE)
  file(
    WRITE ${python_file}
    ""
    "import json\n"
    "import sys\n"
    "with open(sys.argv[1], 'r') as f:\n"
    "    data = json.load(f)\n"
    "    print(data[\"musa_runtime\"][\"version\"])"
    "")

  execute_process(
    COMMAND "python" "${python_file}" ${musa_version_file}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
    RESULT_VARIABLE python_res
    OUTPUT_VARIABLE python_out
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(python_res EQUAL 0)
    set(MUSA_VERSION ${python_out})
  endif()
  string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)" "\\1" MUSA_MAJOR_VERSION
                       "${MUSA_VERSION}")
  string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)" "\\2" MUSA_MINOR_VERSION
                       "${MUSA_VERSION}")
  string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)" "\\3" MUSA_PATCH_VERSION
                       "${MUSA_VERSION}")

  if(NOT MUSA_MAJOR_VERSION)
    set(MUSA_VERSION "???")
    message(WARNING "Cannot find MUSA version in ${MUSA_PATH}/version.json")
  else()
    math(
      EXPR
      MUSA_VERSION
      "${MUSA_MAJOR_VERSION} * 10000 + ${MUSA_MINOR_VERSION} * 100   + ${MUSA_PATCH_VERSION}"
    )
    message(STATUS "Current MUSA version file is ${MUSA_PATH}/version.json.")
    message(
      STATUS
        "Current MUSA version is v${MUSA_MAJOR_VERSION}.${MUSA_MINOR_VERSION}.${MUSA_PATCH_VERSION} "
    )
  endif()
endmacro()
find_musa_version(${MUSA_PATH}/version.json)

list(APPEND MUSA_MCC_FLAGS -Wno-macro-redefined)
list(APPEND MUSA_MCC_FLAGS -Wno-deprecated-copy-with-user-provided-copy)
list(APPEND MUSA_MCC_FLAGS -Wno-pragma-once-outside-header)
list(APPEND MUSA_MCC_FLAGS -Wno-return-type)
list(APPEND MUSA_MCC_FLAGS -Wno-sign-compare)
list(APPEND MUSA_MCC_FLAGS -Wno-overloaded-virtual)
list(APPEND MUSA_MCC_FLAGS -Wno-mismatched-tags)
list(APPEND MUSA_MCC_FLAGS -Wno-pessimizing-move)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-but-set-variable)
list(APPEND MUSA_MCC_FLAGS -Wno-bitwise-instead-of-logical)
list(APPEND MUSA_MCC_FLAGS -Wno-format)
list(APPEND MUSA_MCC_FLAGS -Wno-self-assign)
list(APPEND MUSA_MCC_FLAGS -Wno-literal-conversion)
list(APPEND MUSA_MCC_FLAGS -Wno-literal-range)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-private-field)
list(APPEND MUSA_MCC_FLAGS -Wno-unknown-warning-option)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-variable)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-value)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-local-typedef)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-lambda-capture)
list(APPEND MUSA_MCC_FLAGS -Wno-reorder-ctor)
list(APPEND MUSA_MCC_FLAGS -Wno-braced-scalar-init)
list(APPEND MUSA_MCC_FLAGS -Wno-pass-failed)
list(APPEND MUSA_MCC_FLAGS -Wno-missing-braces)
list(APPEND MUSA_MCC_FLAGS -Wno-dangling-gsl)

if(WITH_CINN)
  list(APPEND MUSA_MCC_FLAGS -std=c++14)
else()
  list(APPEND MUSA_MCC_FLAGS -std=c++17)
endif()

list(APPEND MUSA_MCC_FLAGS --cuda-gpu-arch=mp_22)
list(APPEND MUSA_MCC_FLAGS -U__CUDA__)
# MUSA has compile conflicts of float16.h as platform::float16 overload std::is_floating_point and std::is_integer
list(APPEND MUSA_MCC_FLAGS -D__MUSA_NO_HALF_CONVERSIONS__)

#set(MUSA_VERBOSE_BUILD ON)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  list(APPEND MUSA_MCC_FLAGS -g2)
  list(APPEND MUSA_MCC_FLAGS -O0)
else()
  list(APPEND MUSA_MCC_FLAGS -DRELEASE_MUSA)
  list(APPEND MUSA_MCC_FLAGS -O3)
endif()

set(musa_runtime_library_name musart)
find_library(MUSARTC_LIB ${musa_runtime_library_name} HINTS ${MUSA_PATH}/lib)
message(STATUS "MUSARTC_LIB: ${MUSARTC_LIB}")
