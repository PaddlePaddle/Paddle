<<<<<<< HEAD
add_executable(print_phi_kernels print_phi_kernels.cc)
target_link_libraries(print_phi_kernels phi phi_api_utils)
if(WIN32)
  target_link_libraries(print_phi_kernels shlwapi.lib)
endif()
if(WITH_ROCM)
  target_link_libraries(print_phi_kernels ${ROCM_HIPRTC_LIB})
=======
add_executable(print_pten_kernels print_pten_kernels.cc)
target_link_libraries(print_pten_kernels phi phi_api_utils)
if(WIN32)
  target_link_libraries(print_pten_kernels shlwapi.lib)
endif()
if(WITH_ROCM)
  target_link_libraries(print_pten_kernels ${ROCM_HIPRTC_LIB})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
endif()
