if(NOT WITH_GPU)
  return()
endif()

find_package(PkgConfig)

find_library(
  CUDA_NVRTC_LIB libnvrtc nvrtc
  HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}"
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibNVRTC DEFAULT_MSG CUDA_NVRTC_LIB)

message(STATUS "found NVRTC: ${CUDA_NVRTC_LIB}")

mark_as_advanced(CUDA_NVRTC_LIB)

if(NOT LIBNVRTC_FOUND)
  message(
    FATAL_ERROR
      "Cuda NVRTC Library not found: Specify the LIBNVRTC_LIBRARY_DIR where libnvrtc is located"
  )
endif()
