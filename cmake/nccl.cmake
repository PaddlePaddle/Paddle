if (NOT WITH_GPU)
  return ()
endif()

set(NCCL_ROOT "/usr" CACHE PATH "CUDNN ROOT")
find_path(NCCL_INCLUDE_DIR nccl.h PATHS
        ${NCCL_ROOT} ${NCCL_ROOT}/include
        $ENV{NCCL_ROOT} $ENV{NCCL_ROOT}/include ${CUDA_TOOLKIT_INCLUDE}
        NO_DEFAULT_PATH)

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
  set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(APPEND NCCL_CHECK_LIBRARY_DIRS
        ${NCCL_ROOT}
        ${NCCL_ROOT}/lib64
        ${NCCL_ROOT}/lib
        ${NCCL_ROOT}/lib/${TARGET_ARCH}-linux-gnu
        $ENV{NCCL_ROOT}
        $ENV{NCCL_ROOT}/lib64
        $ENV{NCCL_ROOT}/lib
        /usr/lib)
find_library(NCCL_LIBRARY NAMES libnccl.so libnccl.dylib # libcudnn_static.a
        PATHS ${NCCL_CHECK_LIBRARY_DIRS} ${NCCL_INCLUDE_DIR} ${__libpath_hist}
        NO_DEFAULT_PATH
        DOC "Path to nccl library.")
