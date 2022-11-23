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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
endif()
