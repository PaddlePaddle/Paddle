cinn_cc_library(test_program_builder SRCS program_builder.cc DEPS cinncore)

if(WITH_TESTING)
  include_directories(${CMAKE_SOURCE_DIR}/paddle/cinn/runtime)
  add_subdirectory(benchmark)

  cinn_cc_test(
    test01_elementwise_add_main
    SRCS
    test01_elementwise_add_main.cc
    DEPS
    cinncore
    ARGS
    ${global_test_args})
  cinn_cc_test(test01_elementwise_add_case SRCS test01_elementwise_add_case.cc
               DEPS cinncore)
  add_run_test_dependency(test01_elementwise_add_case
                          test01_elementwise_add_main)

  cinn_cc_test(
    test02_matmul_main
    SRCS
    test02_matmul_main.cc
    DEPS
    cinncore
    ARGS
    ${global_test_args})
  cinn_cc_test(test02_matmul_case SRCS test02_matmul_case.cc DEPS cinncore)
  target_compile_options(test02_matmul_case PRIVATE "-O3")
  add_run_test_dependency(test02_matmul_case test02_matmul_main)

  cinn_cc_test(
    test03_conv_main
    SRCS
    test03_convolution_main.cc
    DEPS
    cinncore
    ARGS
    ${global_test_args})
  cinn_cc_test(test03_conv_case SRCS test03_convolution_case.cc DEPS cinncore)
  target_compile_options(test03_conv_case PRIVATE "-O3")
  add_run_test_dependency(test03_conv_case test03_conv_main)
endif()
