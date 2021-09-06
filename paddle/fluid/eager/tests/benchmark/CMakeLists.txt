cc_library(benchmark_utils SRCS benchmark_utils.cc DEPS ${eager_deps} ${fluid_deps})

cc_test(benchmark_eager_cpu SRCS benchmark_eager_cpu.cc DEPS benchmark_utils ${eager_deps} ${fluid_deps})
cc_test(benchmark_fluid_cpu SRCS benchmark_fluid_cpu.cc DEPS benchmark_utils ${eager_deps} ${fluid_deps})

cc_test(benchmark_eager_cuda SRCS benchmark_eager_cuda.cc DEPS benchmark_utils ${eager_deps} ${fluid_deps})
cc_test(benchmark_fluid_cuda SRCS benchmark_fluid_cuda.cc DEPS benchmark_utils ${eager_deps} ${fluid_deps})
