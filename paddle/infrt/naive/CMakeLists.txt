cc_library(infrt_naive SRCS meta_tensor.cc
        infershaped/infershaped_kernel_launcher.cc
        infershaped/infershaped_registry.cc
        infershaped/infershaped_kernel_launchers.cc
        )

cc_test_tiny(test_infrt_infershape_launchers SRCS 
infershaped/infershape_launchers_test.cc DEPS infrt)
