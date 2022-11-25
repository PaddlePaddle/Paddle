add_executable(print_phi_kernels print_phi_kernels.cc)
target_link_libraries(print_phi_kernels phi phi_api_utils)
if(WIN32)
<<<<<<< HEAD
  target_link_libraries(print_phi_kernels shlwapi.lib)
endif()
if(WITH_ROCM)
  target_link_libraries(print_phi_kernels ${ROCM_HIPRTC_LIB})
=======
  target_link_libraries(print_pten_kernels shlwapi.lib)
endif()
if(WITH_ROCM)
  target_link_libraries(print_pten_kernels ${ROCM_HIPRTC_LIB})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
endif()
