if (NOT INFRT_WITH_PHI)
    return()
endif()

#mlir_tablegen_on(infrt_phi_base DIALECT phi)
add_mlir_dialect(infrt_phi_base phi)
add_mlir_dialect(infrt_phi_tensor phi_dt)
add_mlir_dialect(phi_cpu_kernels phi_cpu_kernel)
add_mlir_dialect(phi_gpu_kernels phi_gpu_kernel)
#mlir_tablegen_on(infrt_phi_tensor)

gather_srcs(infrt_src SRCS
    phi_base.cc infrt_phi_tensor.cc
    phi_kernels.cc)

add_subdirectory(pass)

add_executable(phi-exec phi_exec.cc)
target_link_libraries(phi-exec infrt)
