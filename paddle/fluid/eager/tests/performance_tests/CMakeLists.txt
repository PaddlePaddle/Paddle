cc_library(
  performance_benchmark_utils
  SRCS benchmark_utils.cc
  DEPS ${eager_deps}
       ${fluid_deps}
       ${generated_deps}
       eager_scale
       scale_node
       scale_op
       matmul_v2_op
       dygraph_function)

<<<<<<< HEAD
cc_test(
  test_egr_performance_benchmark_eager_cpu
  SRCS benchmark_eager_cpu.cc
  DEPS performance_benchmark_utils ${eager_deps} ${fluid_deps})
cc_test(
  test_egr_performance_benchmark_fluid_cpu
  SRCS benchmark_fluid_cpu.cc
  DEPS performance_benchmark_utils ${eager_deps} ${fluid_deps})

cc_test(
  test_egr_performance_benchmark_eager_cuda
  SRCS benchmark_eager_cuda.cc
  DEPS performance_benchmark_utils ${eager_deps} ${fluid_deps})
cc_test(
  test_egr_performance_benchmark_fluid_cuda
  SRCS benchmark_fluid_cuda.cc
  DEPS performance_benchmark_utils ${eager_deps} ${fluid_deps})
=======
cc_test_old(
  test_egr_performance_benchmark_eager_cpu
  SRCS
  benchmark_eager_cpu.cc
  DEPS
  performance_benchmark_utils
  ${eager_deps}
  ${fluid_deps})
cc_test_old(
  test_egr_performance_benchmark_fluid_cpu
  SRCS
  benchmark_fluid_cpu.cc
  DEPS
  performance_benchmark_utils
  ${eager_deps}
  ${fluid_deps})

cc_test_old(
  test_egr_performance_benchmark_eager_cuda
  SRCS
  benchmark_eager_cuda.cc
  DEPS
  performance_benchmark_utils
  ${eager_deps}
  ${fluid_deps})
cc_test_old(
  test_egr_performance_benchmark_fluid_cuda
  SRCS
  benchmark_fluid_cuda.cc
  DEPS
  performance_benchmark_utils
  ${eager_deps}
  ${fluid_deps})
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
