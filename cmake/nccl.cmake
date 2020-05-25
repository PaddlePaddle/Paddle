if(NOT WITH_GPU)
    return()
endif()

if(WIN32)
    set(NCCL_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
else(WIN32)
    set(NCCL_ROOT "/usr" CACHE PATH "NCCL ROOT")
endif(WIN32)

find_path(NCCL_INCLUDE_DIR cudnn.h
    PATHS ${NCCL_ROOT} ${NCCL_ROOT}/include
    $ENV{NCCL_ROOT} $ENV{NCCL_ROOT}/include ${CUDA_TOOLKIT_INCLUDE}
    NO_DEFAULT_PATH
)

if(WITH_NCCL)
    file(READ ${NCCL_INCLUDE_DIR}/nccl.h NCCL_VERSION_FILE_CONTENTS)

    string(REGEX MATCH "define NCCL_VERSION_CODE +([0-9]+)"
        NCCL_VERSION "${NCCL_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define NCCL_VERSION_CODE +([0-9]+)" "\\1"
        NCCL_VERSION "${NCCL_VERSION}")

    message(STATUS "Current NCCL version is v${NCCL_VERSION}. ")
endif()
