cc_library(infrt_naive SRCS meta_tensor.cc
        infershape/infershape_launcher.cc
        infershape/infershape_registry.cc
        infershape/infershape_launchers.cc
        )

cc_test_tiny(test_infrt_infershape_launchers SRCS infershape/infershape_launchers_test.cc DEPS infrt)
