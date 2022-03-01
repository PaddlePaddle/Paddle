#mlir_tablegen_on(infrt_phi_base DIALECT phi)
add_mlir_dialect(infrt_phi_base phi)
add_mlir_dialect(infrt_phi_tensor phi_dt)
add_mlir_dialect(phi_cpu_kernels phi_cpu)
add_mlir_dialect(phi_gpu_kernels phi_gpu)

#mlir_tablegen_on(infrt_phi_tensor)

gather_srcs(infrt_src SRCS
    phi_base.cc 
    infrt_phi_tensor.cc
    phi_kernels.cc)
