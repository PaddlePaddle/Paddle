if(NOT (NOT WITH_PYTHON AND ON_INFER))
  if(WITH_CINN)
    set(eager_deps ${eager_deps} python)
  endif()
  cc_library(performance_benchmark_utils SRCS benchmark_utils.cc)
  add_dependencies(
    performance_benchmark_utils
    ${eager_deps}
    ${fluid_deps}
    ${generated_deps}
    eager_scale
    scale_node
    generated_op
    generated_static_op
    dygraph_function
    eager_prim_api)

  paddle_test(test_egr_performance_benchmark_eager_cpu SRCS
              benchmark_eager_cpu.cc DEPS performance_benchmark_utils)
  paddle_test(test_egr_performance_benchmark_fluid_cpu SRCS
              benchmark_fluid_cpu.cc DEPS performance_benchmark_utils)

  if(WITH_GPU)
    paddle_test(test_egr_performance_benchmark_eager_cuda SRCS
                benchmark_eager_cuda.cc DEPS performance_benchmark_utils)
    paddle_test(test_egr_performance_benchmark_fluid_cuda SRCS
                benchmark_fluid_cuda.cc DEPS performance_benchmark_utils)
  endif()

  if(WITH_ONNXRUNTIME AND WIN32)
    # Copy onnxruntime for some c++ test in Windows, since the test will
    # be build only in CI, so suppose the generator in Windows is Ninja.
    copy_onnx(performance_benchmark_utils)
  endif()
endif()
