cc_library(benchmark SRCS benchmark.cc DEPS lod_tensor)
cc_test(benchmark_cpu SRCS benchmark_cpu.cc DEPS benchmark)
