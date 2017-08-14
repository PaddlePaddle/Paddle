if(NOT WITH_GPU)
    return()
endif()

set(CUDNN_ROOT "" CACHE PATH "CUDNN ROOT")
find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS ${CUDNN_ROOT} ${CUDNN_ROOT}/include
    $ENV{CUDNN_ROOT} $ENV{CUDNN_ROOT}/include ${CUDA_TOOLKIT_INCLUDE}
    NO_DEFAULT_PATH
)

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
    set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(APPEND CUDNN_CHECK_LIBRARY_DIRS
    ${CUDNN_ROOT}
    ${CUDNN_ROOT}/lib64
    ${CUDNN_ROOT}/lib
    ${CUDNN_ROOT}/lib/${TARGET_ARCH}-linux-gnu
    $ENV{CUDNN_ROOT}
    $ENV{CUDNN_ROOT}/lib64
    $ENV{CUDNN_ROOT}/lib
    /usr/lib)
find_library(CUDNN_LIBRARY NAMES libcudnn.so libcudnn.dylib # libcudnn_static.a
    PATHS ${CUDNN_CHECK_LIBRARY_DIRS} ${CUDNN_INCLUDE_DIR} ${__libpath_hist}
          NO_DEFAULT_PATH
    DOC "Path to cuDNN library.")


if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    set(CUDNN_FOUND ON)
else()
    set(CUDNN_FOUND OFF)
endif()

if(CUDNN_FOUND)
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

    get_filename_component(CUDNN_LIB_PATH ${CUDNN_LIBRARY} DIRECTORY)

    string(REGEX MATCH "define CUDNN_VERSION +([0-9]+)"
        CUDNN_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_VERSION +([0-9]+)" "\\1"
        CUDNN_VERSION "${CUDNN_VERSION}")

    if("${CUDNN_VERSION}" STREQUAL "2000")
        message(STATUS "Current cuDNN version is v2. ")
    else()
        string(REGEX MATCH "define CUDNN_MAJOR +([0-9]+)" CUDNN_MAJOR_VERSION
            "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR +([0-9]+)" "\\1"
            CUDNN_MAJOR_VERSION "${CUDNN_MAJOR_VERSION}")
        string(REGEX MATCH "define CUDNN_MINOR +([0-9]+)" CUDNN_MINOR_VERSION
            "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR +([0-9]+)" "\\1"
            CUDNN_MINOR_VERSION "${CUDNN_MINOR_VERSION}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL +([0-9]+)"
            CUDNN_PATCHLEVEL_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL +([0-9]+)" "\\1"
            CUDNN_PATCHLEVEL_VERSION "${CUDNN_PATCHLEVEL_VERSION}")

        if(NOT CUDNN_MAJOR_VERSION)
            set(CUDNN_VERSION "???")
        else()
            math(EXPR CUDNN_VERSION
                "${CUDNN_MAJOR_VERSION} * 1000 +
                 ${CUDNN_MINOR_VERSION} * 100 + ${CUDNN_PATCHLEVEL_VERSION}")
        endif()

        message(STATUS "Current cuDNN header is ${CUDNN_INCLUDE_DIR}/cudnn.h. "
            "Current cuDNN version is v${CUDNN_MAJOR_VERSION}. ")

    endif()
endif()
