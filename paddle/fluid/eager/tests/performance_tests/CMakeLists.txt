cc_library(benchmark_utils SRCS benchmark_utils.cc DEPS ${eager_deps} ${fluid_deps} ${generated_deps} scale_op matmul_v2_op )
