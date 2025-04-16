if(WITH_GPU)
  if(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0 AND ${CMAKE_CXX_COMPILER_VERSION}
                                                  VERSION_GREATER 12.0)
    return()
  endif()
endif()

add_executable(print_phi_kernels print_phi_kernels.cc)
target_link_libraries(print_phi_kernels phi common)
if(WIN32)
  target_link_libraries(print_phi_kernels shlwapi.lib)
  # In windows, some operators need to link pir, such as fused_attention
  target_link_libraries(print_phi_kernels pir)
endif()
if(WITH_ROCM)
  target_link_libraries(print_phi_kernels ${ROCM_HIPRTC_LIB})
endif()
