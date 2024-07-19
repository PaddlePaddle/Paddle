add_subdirectory(pir)
core_gather_headers()

gather_srcs(cinnapi_src SRCS graph_compiler_util.cc op_strategy.cc
            compile_error.cc)

cinn_cc_library(pir_compiler SRCS pir_compiler.cc DEPS cinnapi op_dialect_vjp)
