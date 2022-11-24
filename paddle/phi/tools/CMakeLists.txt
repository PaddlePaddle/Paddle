add_executable(print_phi_kernels print_phi_kernels.cc)
target_link_libraries(print_phi_kernels phi phi_api_utils)
if(WIN32)
<<<<<<< HEAD
  target_link_libraries(print_pten_kernels shlwapi.lib)
endif()
if(WITH_ROCM)
  target_link_libraries(print_pten_kernels ${ROCM_HIPRTC_LIB})
=======
  target_link_libraries(print_phi_kernels shlwapi.lib)
endif()
if(WITH_ROCM)
  target_link_libraries(print_phi_kernels ${ROCM_HIPRTC_LIB})
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
endif()
