if (NOT INFRT_WITH_PHI)
    return()
endif()

core_gather_headers()

gather_srcs(infrt_src SRCS
    registry.cc
    dense_tensor_kernels.cc
    context_kernels.cc
)

set(infrt_register_phi_kernels_gen_source_file ${CMAKE_SOURCE_DIR}/paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.cc)
set(infrt_register_phi_kernels_gen_file ${CMAKE_SOURCE_DIR}/tools/infrt/get_phi_kernel_function.sh)
set(wrapped_infermeta_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.h)
set(wrapped_infermeta_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.cc)

add_custom_command(
        OUTPUT ${infrt_register_phi_kernels_gen_source_file}
        COMMAND bash ${infrt_register_phi_kernels_gen_file}
        DEPENDS wrapped_infermeta
        VERBATIM)
add_custom_target(infrt_register_phi_kernel
        COMMAND bash ${infrt_register_phi_kernels_gen_file}
        DEPENDS wrapped_infermeta
        COMMENT "infrt generate ${infrt_register_phi_kernels_gen_source_file}"
        VERBATIM)

cc_library(infrt_naive SRCS infershaped/infershaped_kernel_launcher.cc
        infershaped/infershaped_kernel_launchers.cc
        DEPS phi wrapped_infermeta)

cc_test_tiny(test_infrt_infershape_launchers SRCS
infershaped/infershape_launchers_test.cc DEPS infrt)
