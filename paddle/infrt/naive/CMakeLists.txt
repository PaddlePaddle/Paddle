set(infrt_register_pten_kernels_gen_source_file ${CMAKE_SOURCE_DIR}/paddle/infrt/naive/infershaped/infershaped_kernel_launchers.cc)
set(infrt_register_pten_kernels_gen_file ${CMAKE_SOURCE_DIR}/tools/infrt/get_pten_kernel_function.sh)
set(wrapped_infermeta_header_file ${CMAKE_SOURCE_DIR}/paddle/pten/infermeta/generated.h)
set(wrapped_infermeta_source_file ${CMAKE_SOURCE_DIR}/paddle/pten/infermeta/generated.cc)

add_custom_command(
        OUTPUT ${infrt_register_pten_kernels_gen_source_file}
        COMMAND sh ${infrt_register_pten_kernels_gen_file}
        DEPENDS ${wrapped_infermeta_header_file} ${wrapped_infermeta_source_file}
        COMMENT "infrt generate ${infrt_register_pten_kernels_gen_source_file}"
        VERBATIM)

cc_library(infrt_naive SRCS meta_tensor.cc
        infershaped/infershaped_kernel_launcher.cc
        infershaped/infershaped_registry.cc
        infershaped/infershaped_kernel_launchers.cc
        DEPS infermeta pten wrapped_infermeta
        )

cc_test_tiny(test_infrt_infershape_launchers SRCS 
infershaped/infershape_launchers_test.cc DEPS infrt)
