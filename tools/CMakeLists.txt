cc_library(benchmark SRCS benchmark.cc DEPS lod_tensor op_registry device_context init)
cc_test(benchmark_cpu SRCS benchmark_cpu.cc DEPS benchmark sum_op)
